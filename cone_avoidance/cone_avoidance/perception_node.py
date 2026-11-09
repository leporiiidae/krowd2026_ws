"""LiDAR cone detection + global-frame tracking."""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64MultiArray, Int32
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from .csv_logger import CsvLogger
from .lidar import deskew, detect_cones
from .ros_common import run_node, OdomState, declare_config, stamp_to_sec, yaw_from_quaternion
from .tracker import ConeTracker, RowFilter


def _rpy_matrix(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _quat_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("cone_perception")
        self.cfg = declare_config(self)
        t = self.cfg.topics
        self.tracker = ConeTracker(self.cfg)
        self.rows = RowFilter(self.cfg)
        self.obstacles = []
        self.odom = OdomState(self.cfg)
        self.odom_buffer = deque(maxlen=200)
        self.extrinsic = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.log = CsvLogger(self.cfg.logging.path, "cones", self.cfg.logging.enabled)

        self.create_subscription(Odometry, t.odometry, self.on_odom, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, t.pointcloud, self.on_cloud, qos_profile_sensor_data)
        self.cones_pub = self.create_publisher(Float64MultiArray, t.cones, 10)
        self.raw_pub = self.create_publisher(Float64MultiArray, t.raw_cones, 10)
        self.count_pub = self.create_publisher(Int32, t.lidar_raw_count, 10)
        self.marker_pub = self.create_publisher(MarkerArray, t.cone_markers, 10)
        self.odom_frame = "odom"

    def on_odom(self, msg: Odometry) -> None:
        self.odom.update(msg)
        self.odom_frame = msg.header.frame_id or self.odom_frame
        pose = msg.pose.pose
        self.odom_buffer.append((stamp_to_sec(msg.header.stamp), pose.position.x, pose.position.y,
                                 yaw_from_quaternion(pose.orientation)))

    def _pose_at(self, t: float):
        """Base pose at the scan stamp: interpolate the buffer, else dead-reckon."""
        buf = self.odom_buffer
        for (t0, x0, y0, a0), (t1, x1, y1, a1) in zip(buf, list(buf)[1:]):
            if t0 <= t <= t1 and t1 > t0:
                r = (t - t0) / (t1 - t0)
                da = math.atan2(math.sin(a1 - a0), math.cos(a1 - a0))
                return x0 + r * (x1 - x0), y0 + r * (y1 - y0), a0 + r * da
        return self.odom.base_pose_at(t)

    def _extrinsic(self, frame: str):
        f = self.cfg.frames
        if not f.use_tf or frame == f.base_frame:
            x, y, z, roll, pitch, yaw = f.lidar_xyz_rpy
            return _rpy_matrix(roll, pitch, yaw), np.array([x, y, z])
        if self.extrinsic is None:
            try:
                tf = self.tf_buffer.lookup_transform(f.base_frame, frame, rclpy.time.Time())
            except Exception as error:
                self.get_logger().warn(f"waiting for TF {frame}->{f.base_frame}: {error}",
                                       throttle_duration_sec=2.0)
                return None
            tr = tf.transform.translation
            self.extrinsic = (_quat_matrix(tf.transform.rotation), np.array([tr.x, tr.y, tr.z]))
        return self.extrinsic

    def on_cloud(self, msg: PointCloud2) -> None:
        if not self.odom.valid:
            return
        ext = self._extrinsic(msg.header.frame_id)
        if ext is None:
            return
        p = self.cfg.perception
        names = [f.name for f in msg.fields]
        use_time = p.deskew and p.point_time_field in names
        fields = ["x", "y", "z"] + ([p.point_time_field] if use_time else [])
        cloud = point_cloud2.read_points(msg, field_names=fields, skip_nans=True)
        xyz = np.column_stack([cloud["x"], cloud["y"], cloud["z"]]).astype(float)
        if len(xyz) == 0:
            return
        rot, trans = ext
        xyz = xyz @ rot.T + trans
        if use_time:
            t_rel = cloud[p.point_time_field].astype(float) * p.point_time_scale
            xyz = deskew(xyz, t_rel, self.odom.v, self.odom.yaw_rate)

        local = detect_cones(xyz, self.cfg, self.cfg.frames.ground_z)
        bx, by, byaw = self._pose_at(stamp_to_sec(msg.header.stamp))
        c, s = math.cos(byaw), math.sin(byaw)
        glob = np.column_stack((bx + c * local[:, 0] - s * local[:, 1],
                                by + s * local[:, 0] + c * local[:, 1])) if len(local) else np.zeros((0, 2))
        in_range = np.linalg.norm(local, axis=1) <= p.detect_range if len(local) else np.zeros(0, bool)
        self.tracker.update(glob[in_range])

        self.get_logger().info(
            f"cloud {len(xyz)} pts -> band {int(np.sum((xyz[:, 2] - self.cfg.frames.ground_z >= p.ground_z[0]) & (xyz[:, 2] - self.cfg.frames.ground_z <= p.ground_z[1])))}"
            f" -> cones {len(local)} (in range {int(in_range.sum())})"
            f" -> tracks {len(self.tracker.tracks)} confirmed {len(self.tracker.confirmed())}"
            f" -> obstacles {len(self.obstacles)}", throttle_duration_sec=2.0)
        self.raw_pub.publish(Float64MultiArray(data=glob.reshape(-1).tolist()))
        self.count_pub.publish(Int32(data=int(len(glob))))
        # Only tracks forming an obstacle row (N cones side by side across the road) go on.
        self.obstacles = self.rows.update(self.tracker.confirmed(), byaw, (bx, by))
        rows = []
        for tr in self.obstacles:
            rows.extend([float(tr.track_id), tr.x, tr.y, float(tr.misses)])
        self.cones_pub.publish(Float64MultiArray(data=rows))
        self._markers(msg.header.stamp)
        self.log.write({"t": stamp_to_sec(msg.header.stamp), "base_x": bx, "base_y": by, "base_yaw": byaw,
                        "raw": ";".join(f"{x:.3f},{y:.3f}" for x, y in glob),
                        "tracks": ";".join(f"{t.track_id}:{t.x:.3f},{t.y:.3f},{int(t.confirmed)},{t.misses}"
                                           for t in self.tracker.tracks.values())})

    def _markers(self, stamp) -> None:
        arr = MarkerArray()
        wipe = Marker()
        wipe.action = Marker.DELETEALL
        arr.markers.append(wipe)
        shown = self.tracker.tracks.values() if self.cfg.perception.show_rejected else self.obstacles
        obstacle_ids = {t.track_id for t in self.obstacles}
        for tr in shown:
            m = Marker()
            m.header.frame_id, m.header.stamp = self.odom_frame, stamp
            m.ns, m.id, m.type, m.action = "cones", tr.track_id, Marker.CYLINDER, Marker.ADD
            m.pose.position = Point(
                x=tr.x,
                y=tr.y,
                z=self.odom.ground_z() + self.cfg.perception.cone_height / 2,
            )
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = self.cfg.perception.cone_diameter
            m.scale.z = self.cfg.perception.cone_height
            is_obstacle = tr.track_id in obstacle_ids
            m.color.r, m.color.g, m.color.b = (1.0, 0.5, 0.0) if is_obstacle else (0.6, 0.6, 0.6)
            m.color.a = 0.9 if is_obstacle else 0.3
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def destroy_node(self):
        self.log.close()
        super().destroy_node()


def main(args=None):
    run_node(PerceptionNode)


if __name__ == "__main__":
    main()
