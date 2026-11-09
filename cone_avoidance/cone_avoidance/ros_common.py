"""Shared ROS 2 glue: parameter declaration with live updates, pose helpers."""

from __future__ import annotations

import math
from typing import Any, Dict

from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult

from .params import DEFAULTS, Config, unflatten

_SKIP_SECTIONS = ("sim",)  # simulator-only values are not ROS parameters


def _flatten(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(_flatten(value, name + "."))
        else:
            out[name] = value
    return out


def declare_config(node) -> Config:
    """Declare every parameter on ``node`` and return a Config that tracks edits.

    ``ros2 param set`` rebuilds the Config in place, so planners/controllers that
    hold a reference pick the new value up on their next cycle — no restart.
    """
    flat = _flatten({k: v for k, v in DEFAULTS.items() if k not in _SKIP_SECTIONS})
    descriptor = ParameterDescriptor(dynamic_typing=True)
    values = {}
    for name, default in flat.items():
        values[name] = node.declare_parameter(name, default, descriptor).value
    cfg = Config(unflatten(values))

    def on_set(params):
        changed = {p.name: p.value for p in params if p.name in flat}
        if changed:
            try:
                cfg.apply(unflatten(changed))
            except Exception as error:  # bad type/value: refuse instead of crashing the loop
                return SetParametersResult(successful=False, reason=str(error))
            node.get_logger().info(f"parameters updated: {changed}")
        return SetParametersResult(successful=True)

    node.add_on_set_parameters_callback(on_set)
    return cfg


def yaw_from_quaternion(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


class OdomState:
    """Latest odometry, with the lever arm to the front axle applied."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.valid = False
        self.stamp = 0.0
        self.x = self.y = self.z = self.yaw = self.v = self.yaw_rate = 0.0

    def update(self, msg) -> None:
        pose = msg.pose.pose
        self.x, self.y, self.z = pose.position.x, pose.position.y, pose.position.z
        self.yaw = yaw_from_quaternion(pose.orientation)
        self.v = msg.twist.twist.linear.x
        self.yaw_rate = msg.twist.twist.angular.z
        self.stamp = stamp_to_sec(msg.header.stamp)
        self.valid = True

    def ground_z(self) -> float:
        """Height of the road surface in the odometry frame, at the vehicle.

        RViz markers must sit on the ground, and the odometry frame's z is not zero on
        this vehicle — drawing at z = 0 puts the cones and the lane tens of metres away
        from the cloud, which looks exactly like "nothing was detected".
        """
        return self.z + self.cfg.frames.ground_z

    def base_pose_at(self, t: float):
        """Dead-reckon the base pose to time ``t`` (bridges the gaps between fixes)."""
        dt = min(max(t - self.stamp, -0.2), 0.2)
        yaw_mid = self.yaw + 0.5 * self.yaw_rate * dt
        return (self.x + self.v * math.cos(yaw_mid) * dt,
                self.y + self.v * math.sin(yaw_mid) * dt,
                self.yaw + self.yaw_rate * dt)

    def front_axle_at(self, t: float):
        x, y, yaw = self.base_pose_at(t)
        f = self.cfg.frames
        c, s = math.cos(yaw), math.sin(yaw)
        return (x + c * f.front_axle_x - s * f.front_axle_y,
                y + s * f.front_axle_x + c * f.front_axle_y, yaw)


def run_node(factory) -> None:
    """Spin one node and shut down quietly on Ctrl-C / launch SIGINT."""
    import rclpy
    from rclpy.executors import ExternalShutdownException

    rclpy.init()
    node = None
    try:
        node = factory()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            try:
                node.destroy_node()
            except BaseException:
                pass  # second SIGINT / context already torn down by the signal handler
        rclpy.try_shutdown()
