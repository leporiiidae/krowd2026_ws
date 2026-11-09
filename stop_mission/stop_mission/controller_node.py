"""Standalone Pure Pursuit controller for the cone mission."""

from __future__ import annotations

import math

from nav_msgs.msg import Odometry, Path
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Bool, Float32, String
from visualization_msgs.msg import Marker


class PurePursuitControllerNode(Node):
    """Track the local base_link path and publish ros_can_bridge physical commands."""

    def __init__(self) -> None:
        super().__init__("stop_mission_pure_pursuit")
        defaults = {
            "path_topic": "/stop_mission/path",
            "end_distance_topic": "/stop_mission/end_distance",
            "odometry_topic": "/ekf/odometry",
            "done_topic": "/stop_mission/done",
            "start_topic": "/stop_mission/start",
            "angle_topic": "/target_angle",
            "speed_topic": "/target_speed",
            "brake_topic": "/brake",
            "status_topic": "/stop_mission/control_status",
            "lookahead_marker_topic": "/stop_mission/lookahead_marker",
            "auto_start": True,
            "control_rate": 20.0,
            "lookahead_distance": 3.0,
            "wheelbase": 1.34,
            "steering_sign": 1.0,
            "max_steering_angle_deg": 25.0,
            "target_speed_mps": 4.0,
            "braking_trigger_distance_m": 3.0,
            "braking_confirmation_frames": 10,
            "stop_speed_tolerance_mps": 0.20,
            "done_confirmation_cycles": 5,
            "path_timeout": 0.5,
            "end_distance_timeout": 0.5,
            "time_rewind_tolerance_sec": 0.1,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self.started = bool(self.parameter("auto_start"))
        self.done = False
        self.path: Path | None = None
        self.last_path_time = None
        self.end_distance: float | None = None
        self.last_end_distance_time = None
        self.measured_speed: float | None = None
        self.braking = False
        self.braking_confirmation_count = 0
        self.done_cycles = 0
        self.last_odometry_stamp_sec: float | None = None

        self.create_subscription(
            Path, str(self.parameter("path_topic")), self.path_callback, 10
        )
        self.create_subscription(
            Float32,
            str(self.parameter("end_distance_topic")),
            self.end_distance_callback,
            10,
        )
        self.create_subscription(
            Odometry,
            str(self.parameter("odometry_topic")),
            self.odometry_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool, str(self.parameter("start_topic")), self.start_callback, 10
        )
        self.angle_pub = self.create_publisher(
            Float32, str(self.parameter("angle_topic")), 10
        )
        self.speed_pub = self.create_publisher(
            Float32, str(self.parameter("speed_topic")), 10
        )
        self.brake_pub = self.create_publisher(
            Bool, str(self.parameter("brake_topic")), 10
        )
        self.done_pub = self.create_publisher(
            Bool, str(self.parameter("done_topic")), 10
        )
        self.status_pub = self.create_publisher(
            String, str(self.parameter("status_topic")), 10
        )
        self.lookahead_marker_pub = self.create_publisher(
            Marker, str(self.parameter("lookahead_marker_topic")), 10
        )
        self.create_timer(
            1.0 / max(float(self.parameter("control_rate")), 1.0), self.control_once
        )
        self.publish_stop("INITIALIZING")

    def parameter(self, name: str):
        return self.get_parameter(name).value

    def path_callback(self, message: Path) -> None:
        self.path = message
        self.last_path_time = self.get_clock().now()

    def end_distance_callback(self, message: Float32) -> None:
        if math.isfinite(message.data):
            self.end_distance = float(message.data)
            self.last_end_distance_time = self.get_clock().now()
        else:
            self.braking_confirmation_count = 0

    def odometry_callback(self, message: Odometry) -> None:
        stamp = message.header.stamp
        stamp_sec = float(stamp.sec) + float(stamp.nanosec) * 1.0e-9
        tolerance = float(self.parameter("time_rewind_tolerance_sec"))
        if (
            self.last_odometry_stamp_sec is not None
            and stamp_sec < self.last_odometry_stamp_sec - tolerance
        ):
            self.get_logger().info(
                "Odometry time moved backwards; resetting Pure Pursuit replay state"
            )
            self.reset_runtime_state()
        self.last_odometry_stamp_sec = stamp_sec
        velocity = message.twist.twist.linear
        self.measured_speed = math.hypot(float(velocity.x), float(velocity.y))

    def start_callback(self, message: Bool) -> None:
        self.started = bool(message.data)
        if self.started:
            self.reset_runtime_state()

    def reset_runtime_state(self) -> None:
        self.done = False
        self.braking = False
        self.braking_confirmation_count = 0
        self.done_cycles = 0
        self.path = None
        self.last_path_time = None
        self.end_distance = None
        self.last_end_distance_time = None

    def control_once(self) -> None:
        if not self.started:
            self.publish_stop("WAIT_START")
            return
        if self.done:
            self.publish_stop("MISSION_DONE")
            self.done_pub.publish(Bool(data=True))
            return
        self.done_pub.publish(Bool(data=False))
        if self.braking:
            self.publish_braking("BRAKING_LATCHED")
            return
        if (
            self.path is None
            or not self.path.poses
            or self.path_is_stale()
            or self.end_distance_is_stale()
        ):
            self.braking_confirmation_count = 0
            self.publish_stop("NO_FRESH_PATH_OR_ENDPOINT")
            return
        target = self.select_target()
        if target is None:
            self.braking_confirmation_count = 0
            self.publish_stop("NO_LOOKAHEAD")
            return

        x, y = target
        squared_distance = max(x * x + y * y, 1.0e-6)
        steering = math.atan2(
            2.0 * float(self.parameter("wheelbase")) * y, squared_distance
        )
        steering *= float(self.parameter("steering_sign"))
        limit = math.radians(float(self.parameter("max_steering_angle_deg")))
        steering = max(-limit, min(limit, steering))
        remaining = max(0.0, self.end_distance)
        required_frames = max(
            1, int(self.parameter("braking_confirmation_frames"))
        )
        trigger_distance = float(self.parameter("braking_trigger_distance_m"))
        self.braking_confirmation_count = self.next_braking_confirmation_count(
            remaining,
            trigger_distance,
            self.braking_confirmation_count,
        )
        if self.braking_confirmation_count >= required_frames:
            self.braking = True
            self.publish_braking(
                f"BRAKING_TRIGGER end={remaining:.2f}m "
                f"threshold={trigger_distance:.2f}m "
                f"frames={self.braking_confirmation_count}/{required_frames}"
            )
            return

        speed = self.target_speed_mps()
        speed_kph = speed * 3.6
        self.publish_lookahead_marker(target)
        self.angle_pub.publish(Float32(data=float(steering)))
        self.speed_pub.publish(Float32(data=float(speed)))
        self.brake_pub.publish(Bool(data=False))
        self.publish_status(
            f"TRACKING target=({x:.2f},{y:.2f}) end={remaining:.2f}m "
            f"stop_frames={self.braking_confirmation_count}/{required_frames} "
            f"steer={steering:.3f}rad speed={speed_kph:.1f}km/h ({speed:.2f}m/s)"
        )

    def target_speed_mps(self) -> float:
        """Return the non-negative target speed configured in the YAML SI unit."""
        return max(float(self.parameter("target_speed_mps")), 0.0)

    @staticmethod
    def next_braking_confirmation_count(
        end_distance_m: float, trigger_distance_m: float, current_count: int
    ) -> int:
        if end_distance_m <= trigger_distance_m:
            return current_count + 1
        return 0

    def select_target(self) -> tuple[float, float] | None:
        lookahead = float(self.parameter("lookahead_distance"))
        closest = None
        closest_error = math.inf
        for pose in self.path.poses:
            x = float(pose.pose.position.x)
            y = float(pose.pose.position.y)
            if x <= 0.0:
                continue
            distance = math.hypot(x, y)
            if distance >= lookahead:
                return x, y
            error = abs(distance - lookahead)
            if error < closest_error:
                closest = (x, y)
                closest_error = error
        return closest

    def path_is_stale(self) -> bool:
        if self.last_path_time is None:
            return True
        age = (self.get_clock().now() - self.last_path_time).nanoseconds * 1.0e-9
        return age < 0.0 or age > float(self.parameter("path_timeout"))

    def end_distance_is_stale(self) -> bool:
        if self.end_distance is None or self.last_end_distance_time is None:
            return True
        age = (
            self.get_clock().now() - self.last_end_distance_time
        ).nanoseconds * 1.0e-9
        return age < 0.0 or age > float(self.parameter("end_distance_timeout"))

    def current_speed(self) -> float:
        return self.measured_speed if self.measured_speed is not None else 0.0

    def update_done_confirmation(self) -> None:
        stopped = self.current_speed() <= float(
            self.parameter("stop_speed_tolerance_mps")
        )
        if stopped:
            self.done_cycles += 1
        else:
            self.done_cycles = 0
        if self.done_cycles >= int(self.parameter("done_confirmation_cycles")):
            self.done = True
            self.done_pub.publish(Bool(data=True))

    def publish_braking(self, reason: str) -> None:
        self.publish_lookahead_marker(None)
        self.angle_pub.publish(Float32(data=0.0))
        self.speed_pub.publish(Float32(data=0.0))
        self.brake_pub.publish(Bool(data=True))
        self.update_done_confirmation()
        actual_speed = self.current_speed()
        self.publish_status(
            f"{reason} actual_speed={actual_speed * 3.6:.1f}km/h "
            f"({actual_speed:.2f}m/s)"
        )

    def publish_stop(self, reason: str) -> None:
        self.publish_lookahead_marker(None)
        self.angle_pub.publish(Float32(data=0.0))
        self.speed_pub.publish(Float32(data=0.0))
        self.brake_pub.publish(Bool(data=True))
        self.publish_status(reason)

    def publish_status(self, value: str) -> None:
        self.status_pub.publish(String(data=value))

    def publish_lookahead_marker(
        self, target: tuple[float, float] | None
    ) -> None:
        """Show the exact Pure Pursuit target selected from the current path."""
        marker = Marker()
        marker.header.frame_id = (
            self.path.header.frame_id
            if self.path is not None and self.path.header.frame_id
            else "base_link"
        )
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "pure_pursuit_lookahead"
        marker.id = 0
        marker.pose.orientation.w = 1.0
        if target is None:
            marker.action = Marker.DELETE
        else:
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(target[0])
            marker.pose.position.y = float(target[1])
            marker.pose.position.z = 0.25
            marker.scale.x = 0.50
            marker.scale.y = 0.50
            marker.scale.z = 0.50
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
        self.lookahead_marker_pub.publish(marker)

    def destroy_node(self):
        if rclpy.ok():
            self.publish_stop("SHUTDOWN")
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PurePursuitControllerNode()
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
