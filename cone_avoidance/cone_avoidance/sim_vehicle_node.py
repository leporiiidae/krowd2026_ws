"""Hardware-free stand-in for the vehicle: bicycle plant + synthetic LiDAR / YOLO.

Replaces can_node + sensors so the full ROS graph (perception -> mission ->
control) can be exercised on a desk:

    ros2 launch cone_avoidance sim.launch.py
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, Float32, Header, Int32
from tf2_ros import TransformBroadcaster

from .can_codec import from_cmd_units
from .lidar import synth_scan
from .params import DEFAULTS
from .ros_common import run_node, declare_config

LIDAR_X, LIDAR_Z = 0.6, 0.8          # sensor pose in base_link (rear axle)


class SimVehicleNode(Node):
    def __init__(self):
        super().__init__("cone_sim_vehicle")
        self.cfg = declare_config(self)
        self.declare_parameter("sim.cones", DEFAULTS["sim"]["cones"])
        self.cones = np.asarray(self.get_parameter("sim.cones").value, dtype=float).reshape(-1, 2)
        t = self.cfg.topics
        self.x = self.y = self.yaw = self.v = self.delta = 0.0
        self.steer_cmd = self.speed_cmd = 0.0
        self.dt = 0.02
        self.queue = deque([0.0] * 3)
        self.step_count = 0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            Float32, t.steer_cmd,
            lambda m: setattr(self, "steer_cmd", from_cmd_units(float(m.data), 0.0, self.cfg)[0]), 10)
        self.create_subscription(
            Float32, t.speed_cmd,
            lambda m: setattr(self, "speed_cmd", from_cmd_units(0.0, float(m.data), self.cfg)[1]), 10)
        self.odom_pub = self.create_publisher(Odometry, t.odometry, qos_profile_sensor_data)
        self.cloud_pub = self.create_publisher(PointCloud2, t.pointcloud, qos_profile_sensor_data)
        self.yolo_pub = self.create_publisher(Int32, t.yolo_count, 10)
        self.fb_pub = self.create_publisher(Float32, t.steer_feedback, 10)
        self.ws_pub = self.create_publisher(Float32, t.wheel_speed, 10)
        self.zero_pub = self.create_publisher(Bool, t.zero_done, latched)
        self.zero_pub.publish(Bool(data=True))
        self.tf_pub = TransformBroadcaster(self)   # RViz needs odom -> base_link for the cloud
        self.create_timer(self.dt, self.tick)

    def tick(self):
        cfg = self.cfg
        wb = cfg.vehicle.wheelbase
        self.queue.append(self.steer_cmd)
        target = self.queue.popleft()
        k = cfg.max_steer_rate / 0.125
        rate = max(-cfg.max_steer_rate, min(cfg.max_steer_rate, k * (target - self.delta)))
        self.delta += rate * self.dt
        self.v += max(-0.8 * self.dt, min(0.8 * self.dt, self.speed_cmd - self.v))
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        yaw_rate = self.v / wb * math.tan(self.delta)
        self.yaw += yaw_rate * self.dt
        self.step_count += 1

        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, "odom", "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.x, self.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = self.v, yaw_rate
        self.odom_pub.publish(odom)
        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = "base_link"
        tf.transform.translation.x, tf.transform.translation.y = self.x, self.y
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf_pub.sendTransform(tf)
        self.fb_pub.publish(Float32(data=float(self.delta)))
        self.ws_pub.publish(Float32(data=float(self.v)))
        if self.step_count % 5 == 0:
            self.cloud_pub.publish(self._cloud(stamp))
        if self.step_count % 6 == 0:
            self.yolo_pub.publish(Int32(data=self._yolo()))

    def _local_cones(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = self.cones[:, 0] - self.x, self.cones[:, 1] - self.y
        return np.column_stack((c * dx + s * dy, -s * dx + c * dy))

    def _cloud(self, stamp) -> PointCloud2:
        local = self._local_cones() - np.array([LIDAR_X, 0.0])
        local = local[local[:, 0] > 0.3]
        # synth_scan works about the ground plane; shift into base_link afterwards.
        ground_z = self.cfg.frames.ground_z
        scan = synth_scan(local, self.cfg.cone_radius, LIDAR_Z - ground_z,
                          self.cfg.perception.cone_height)
        scan[:, 0] += LIDAR_X
        scan[:, 2] += ground_z
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate("xyz")]
        return point_cloud2.create_cloud(Header(stamp=stamp, frame_id="base_link"), fields,
                                         scan.tolist())

    def _yolo(self) -> int:
        local = self._local_cones()
        ahead = local[:, 0] - self.cfg.vehicle.wheelbase
        ang = np.degrees(np.arctan2(local[:, 1], np.maximum(ahead, 1e-3)))
        return int(np.sum((ahead > 0.3) & (ahead < 12.0) & (np.abs(ang) < 35.0)))


def main(args=None):
    run_node(SimVehicleNode)


if __name__ == "__main__":
    main()
