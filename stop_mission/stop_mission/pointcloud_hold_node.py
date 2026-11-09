"""Repeat the last LiDAR scan on a visualization-only topic while ROS time is paused."""

from __future__ import annotations

import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2


class PointCloudHoldNode(Node):
    """Keep RViz populated during rosbag pause without feeding stale scans to control."""

    def __init__(self) -> None:
        super().__init__("stop_mission_pointcloud_hold")
        self.declare_parameter("input_topic", "/ouster/points")
        self.declare_parameter("output_topic", "/stop_mission/held_points")
        self.declare_parameter("republish_rate_hz", 2.0)

        rate = float(self.get_parameter("republish_rate_hz").value)
        if rate <= 0.0:
            raise ValueError("republish_rate_hz must be greater than zero")

        self.last_cloud: PointCloud2 | None = None
        self.publisher = self.create_publisher(
            PointCloud2,
            str(self.get_parameter("output_topic").value),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PointCloud2,
            str(self.get_parameter("input_topic").value),
            self.cloud_callback,
            qos_profile_sensor_data,
        )
        # A SYSTEM_TIME timer keeps running when rosbag --clock is paused.
        self.wall_clock = Clock(clock_type=ClockType.SYSTEM_TIME)
        self.create_timer(1.0 / rate, self.publish_last_cloud, clock=self.wall_clock)
        self.get_logger().info(
            "%s -> %s (live scans relayed immediately; last scan repeated at "
            "%.1f Hz while paused)"
            % (
                self.get_parameter("input_topic").value,
                self.get_parameter("output_topic").value,
                rate,
            )
        )

    def cloud_callback(self, message: PointCloud2) -> None:
        self.last_cloud = message
        # Keep the RViz cloud synchronized with the perception markers while data
        # is flowing. The wall timer below only exists to preserve that last cloud
        # when rosbag simulation time and sensor publication are paused.
        self.publisher.publish(message)

    def publish_last_cloud(self) -> None:
        if self.last_cloud is not None:
            self.publisher.publish(self.last_cloud)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PointCloudHoldNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
