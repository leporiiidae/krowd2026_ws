"""Repeat the last LiDAR scan on a visualization-only topic while ROS time is paused.

Perception and control never subscribe to this topic: pausing a rosbag must not feed
stale scans into the mission, it should only keep the RViz cloud on screen.
"""

from __future__ import annotations

from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2

from .ros_common import declare_config, run_node


class PointCloudHoldNode(Node):
    def __init__(self) -> None:
        super().__init__("cone_pointcloud_hold")
        self.cfg = declare_config(self)
        self.declare_parameter("republish_rate_hz", 2.0)
        rate = max(float(self.get_parameter("republish_rate_hz").value), 0.1)
        self.last_cloud: PointCloud2 | None = None

        self.publisher = self.create_publisher(
            PointCloud2, self.cfg.topics.held_points,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE))
        self.create_subscription(
            PointCloud2, self.cfg.topics.pointcloud, self.on_cloud, qos_profile_sensor_data)
        # SYSTEM_TIME: this timer keeps ticking while `ros2 bag play --clock` is paused.
        self.create_timer(1.0 / rate, self.republish, clock=Clock(clock_type=ClockType.SYSTEM_TIME))

    def on_cloud(self, message: PointCloud2) -> None:
        self.last_cloud = message
        self.publisher.publish(message)   # live scans pass straight through

    def republish(self) -> None:
        if self.last_cloud is not None:
            self.publisher.publish(self.last_cloud)


def main(args=None):
    run_node(PointCloudHoldNode)


if __name__ == "__main__":
    main()
