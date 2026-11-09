"""FSM + path planning node (10 Hz)."""

from __future__ import annotations

import json
import math

import numpy as np
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Bool, Float32, Float64MultiArray, Int32, String
from visualization_msgs.msg import Marker, MarkerArray

from .csv_logger import CsvLogger
from .mission import Mission
from .ros_common import run_node, OdomState, declare_config


class MissionNode(Node):
    def __init__(self):
        super().__init__("cone_mission")
        self.cfg = declare_config(self)
        t = self.cfg.topics
        self.mission = Mission(self.cfg)
        self.odom = OdomState(self.cfg)
        self.odom_frame = "odom"
        self.tracks = []
        self.rtk_status = None
        self.rtk_time = -math.inf
        self.faults = {"control": False, "can": False}
        self.log = CsvLogger(self.cfg.logging.path, "plan", self.cfg.logging.enabled)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

        self.create_subscription(Odometry, t.odometry, self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, t.cones, self.on_cones, 10)
        self.create_subscription(Float64MultiArray, t.raw_cones, self.on_raw, 10)
        self.create_subscription(Int32, t.yolo_count, self.on_yolo, 10)
        self.create_subscription(Bool, t.zero_done, self.on_zero_done, latched)
        self.create_subscription(Bool, t.start, self.on_start, 10)
        self.create_subscription(NavSatFix, t.navsat, self.on_navsat, qos_profile_sensor_data)
        self.create_subscription(String, t.control_status, lambda m: self.on_status("control", m), 10)
        self.create_subscription(String, t.can_status, lambda m: self.on_status("can", m), 10)
        self.path_pub = self.create_publisher(Float64MultiArray, t.path, 10)
        self.path_viz_pub = self.create_publisher(Path, t.path + "_viz", 10)
        self.speed_pub = self.create_publisher(Float32, t.speed_target, 10)
        self.status_pub = self.create_publisher(String, t.mission_status, 10)
        self.viz_pub = self.create_publisher(MarkerArray, t.mission_markers, 10)
        self.create_timer(0.1, self.tick)

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        self.odom.update(msg)
        self.odom_frame = msg.header.frame_id or self.odom_frame

    def on_cones(self, msg):
        rows = np.asarray(msg.data, dtype=float).reshape(-1, 4)
        self.tracks = [(int(r[0]), r[1], r[2]) for r in rows]

    def on_raw(self, msg):
        self.mission.on_lidar_frame(np.asarray(msg.data, dtype=float).reshape(-1, 2), self.now())

    def on_yolo(self, msg):
        self.mission.on_yolo(int(msg.data), self.now())

    def on_zero_done(self, msg):
        if msg.data:
            self.mission.on_zero_done()

    def on_start(self, msg):
        if msg.data:
            self.mission.on_start()

    def on_navsat(self, msg):
        self.rtk_status = int(msg.status.status)
        self.rtk_time = self.now()

    def on_status(self, source, msg):
        try:
            self.faults[source] = bool(json.loads(msg.data).get("fault", False))
        except ValueError:
            pass

    def tick(self):
        if not self.odom.valid:
            return
        now = self.now()
        r = self.cfg.rtk
        # No NavSatFix ever received: the topic is probably not wired, do not degrade.
        if r.enabled and self.rtk_status is not None:
            fresh = now - self.rtk_time <= r.timeout
            self.mission.rtk_ok = fresh and self.rtk_status >= r.fix_status_min
        self.mission.external_fault = any(self.faults.values())

        x_f, y_f, yaw = self.odom.front_axle_at(now)
        bx, by, _ = self.odom.base_pose_at(now)
        status = self.mission.update(now, x_f, y_f, yaw, self.odom.v, self.tracks, (bx, by))
        for event in status["events"]:
            self.get_logger().info(f"[{status['state']}] {event}")

        if self.mission.ref is not None:
            ref = self.mission.ref
            self.path_pub.publish(Float64MultiArray(
                data=[ref.origin_x, ref.origin_y, ref.heading, *self.mission.path.to_list()]))
            self._publish_viz(status["s"])
            self._publish_markers(status)
        self.speed_pub.publish(Float32(data=float(self.mission.speed_command())))
        status["faults"] = [k for k, v in self.faults.items() if v]
        self.status_pub.publish(String(data=json.dumps(status)))
        row = dict(status)
        row["events"] = "|".join(status["events"])
        row["flags"] = "|".join(status["flags"])
        row["faults"] = "|".join(status["faults"])
        self.log.write(row)

    def _publish_viz(self, s_now):
        ref = self.mission.ref
        msg = Path()
        msg.header.frame_id = self.odom_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        ground_z = self.odom.ground_z() + 0.05   # 5 cm above the road so it is not z-fought
        stations = np.arange(s_now - 2.0, s_now + 20.0, 0.2)
        d, d1, _ = self.mission.path.eval_array(stations)
        for s, dd, slope in zip(stations, d, d1):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x, pose.pose.position.y = ref.to_global(float(s), float(dd))
            pose.pose.position.z = ground_z
            half = 0.5 * (ref.heading + math.atan(float(slope)))
            pose.pose.orientation.z, pose.pose.orientation.w = math.sin(half), math.cos(half)
            msg.poses.append(pose)
        self.path_viz_pub.publish(msg)

    def _marker(self, ns, mid, mtype, scale, rgba):
        m = Marker()
        m.header.frame_id = self.odom_frame
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
        m.scale.x = m.scale.y = m.scale.z = scale
        m.pose.orientation.w = 1.0
        m.color.r, m.color.g, m.color.b, m.color.a = rgba
        return m

    def _publish_markers(self, status) -> None:
        """Lane lines, the collision rectangle and a state label, in the odom frame."""
        ref = self.mission.ref
        cfg = self.cfg
        arr = MarkerArray()
        s_now = status["s"]
        half = cfg.road.width / 2.0
        ground_z = self.odom.ground_z() + 0.05
        # Draw the lane the planner actually uses: once a group is in view the lane is
        # anchored on it (its outer edge = the lane line), not on the reference line.
        # Faint until then, because the reference line alone can be a metre off out there.
        lane_c = self.mission.lane_c
        alpha = 0.9 if lane_c is not None else 0.3
        centre = lane_c if lane_c is not None else 0.0

        for name, side, rgb in (("centre_line", +1, (1.0, 0.9, 0.2)),
                                ("solid_line", -1, (1.0, 1.0, 1.0))):
            m = self._marker(name, 0, Marker.LINE_STRIP, 0.05, (*rgb, alpha))
            for s in (s_now - 10.0, s_now + 30.0):
                x, y = ref.to_global(s, centre + side * half)
                m.points.append(Point(x=x, y=y, z=ground_z))
            arr.markers.append(m)

        x_f, y_f, yaw = self.odom.front_axle_at(self.now())
        c, sn = math.cos(yaw), math.sin(yaw)
        x_r, y_r = x_f - cfg.vehicle.wheelbase * c, y_f - cfg.vehicle.wheelbase * sn
        hw = cfg.collision_width / 2.0
        body = self._marker("vehicle", 0, Marker.LINE_STRIP, 0.05, (0.2, 0.8, 1.0, 0.9))
        corners = [(-cfg.vehicle.rear_overhang, -hw), (cfg.rear_axle_to_front, -hw),
                   (cfg.rear_axle_to_front, hw), (-cfg.vehicle.rear_overhang, hw),
                   (-cfg.vehicle.rear_overhang, -hw)]
        for px, py in corners:
            body.points.append(Point(x=x_r + c * px - sn * py,
                                     y=y_r + sn * px + c * py,
                                     z=ground_z))
        arr.markers.append(body)

        label = self._marker("state", 0, Marker.TEXT_VIEW_FACING, 0.5, (1.0, 1.0, 1.0, 0.9))
        label.pose.position = Point(x=x_f, y=y_f, z=self.odom.ground_z() + 1.5)
        clr = status.get("clearance_rect")
        label.text = (f"{status['state']}  v={status['v_cmd']:.2f}  d={status['d']:+.2f}"
                      + (f"  clr={clr:.2f}" if clr == clr else ""))
        arr.markers.append(label)
        self.viz_pub.publish(arr)

    def destroy_node(self):
        self.log.close()
        super().destroy_node()


def main(args=None):
    run_node(MissionNode)


if __name__ == "__main__":
    main()
