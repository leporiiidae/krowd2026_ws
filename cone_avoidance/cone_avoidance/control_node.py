"""Stanley controller and ROS control node."""

from __future__ import annotations

from .mission_node import OffsetPath, ReferenceLine, wrap

"""Stanley lateral controller on the front axle, with latency compensation."""


import math
from dataclasses import dataclass



@dataclass
class StanleyOutput:
    delta: float        # wheel angle command, LEFT positive
    e_y: float
    theta_e: float
    s: float            # front-axle Frenet of the PREDICTED pose
    d: float
    d_path: float
    delta_raw: float    # before clip / rate limit


class StanleyController:
    def __init__(self, cfg):
        self.cfg = cfg
        self.delta_prev = 0.0

    def reset(self) -> None:
        self.delta_prev = 0.0

    def predict(self, x_f: float, y_f: float, yaw: float, v: float, t_lat: float):
        """Front-axle pose ``t_lat`` seconds ahead, holding the previous wheel angle.

        The rear axle moves along the body yaw; the front axle is rebuilt from it,
        so the prediction stays consistent with the bicycle model.
        """
        wb = self.cfg.vehicle.wheelbase
        x_r = x_f - wb * math.cos(yaw)
        y_r = y_f - wb * math.sin(yaw)
        yaw_rate = v / wb * math.tan(self.delta_prev)
        yaw_mid = yaw + 0.5 * yaw_rate * t_lat
        x_r += v * math.cos(yaw_mid) * t_lat
        y_r += v * math.sin(yaw_mid) * t_lat
        yaw_p = yaw + yaw_rate * t_lat
        return x_r + wb * math.cos(yaw_p), y_r + wb * math.sin(yaw_p), yaw_p

    def step(self, ref: ReferenceLine, path: OffsetPath, x_f: float, y_f: float,
             yaw: float, v: float, dt: float) -> StanleyOutput:
        cfg = self.cfg
        c = cfg.control
        x_p, y_p, yaw_p = self.predict(x_f, y_f, yaw, v, c.latency_compensation)
        s, d = ref.to_frenet(x_p, y_p)
        d_path, d1, _ = path.eval(s)
        path_yaw = ref.heading + math.atan(d1)

        e_y = d - d_path
        theta_e = wrap(path_yaw - yaw_p)
        # k_soft keeps the cross-track term bounded as v -> 0 (launch, slow-down).
        raw = c.k_heading * theta_e + math.atan2(-c.k_stanley * e_y, c.k_soft + abs(v))
        delta = min(max(raw, -cfg.max_wheel_steer), cfg.max_wheel_steer)
        max_step = cfg.max_steer_rate * dt
        delta = min(max(delta, self.delta_prev - max_step), self.delta_prev + max_step)
        self.delta_prev = delta
        return StanleyOutput(delta, e_y, theta_e, s, d, d_path, raw)


MPS_TO_KMH = 3.6


def wheel_to_output(wheel_rad: float, cfg) -> float:
    """Convert road-wheel angle to the output-shaft angle expected by /target_angle."""
    out = wheel_rad * cfg.steering.output_ratio
    limit = cfg.steering.output_limit_rad
    return max(-limit, min(limit, out))


def to_cmd_units(wheel_rad: float, speed_mps: float, cfg):
    """Convert controller units to the external ROS-to-CAN bridge topic units."""
    if cfg.control.steer_cmd_unit == "wheel":
        angle = max(-cfg.max_wheel_steer, min(cfg.max_wheel_steer, wheel_rad))
    else:
        angle = wheel_to_output(wheel_rad, cfg)
    speed = speed_mps * MPS_TO_KMH if cfg.control.speed_cmd_unit == "kmh" else speed_mps
    return angle, speed


def from_cmd_units(angle: float, speed: float, cfg):
    """Inverse conversion used only by the synthetic ROS vehicle."""
    wheel = angle if cfg.control.steer_cmd_unit == "wheel" else angle / cfg.steering.output_ratio
    speed_mps = speed / MPS_TO_KMH if cfg.control.speed_cmd_unit == "kmh" else speed
    return wheel, speed_mps
"""Stanley control node (50 Hz) — also the per-cycle logger."""


import json
import math

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32, Float64MultiArray, String

from .csv_logger import CsvLogger
from .ros_common import run_node, OdomState, declare_config


class ControlNode(Node):
    def __init__(self):
        super().__init__("cone_control")
        self.cfg = declare_config(self)
        t = self.cfg.topics
        self.stanley = StanleyController(self.cfg)
        self.odom = OdomState(self.cfg)
        self.ref = None
        self.path = OffsetPath()
        self.speed_target = 0.0
        self.mission = {}
        self.steer_fb = math.nan
        self.wheel_speed = math.nan
        self.track_err_since = None
        self.odom_wall = -math.inf
        self.log = CsvLogger(self.cfg.logging.path, "control", self.cfg.logging.enabled)

        self.create_subscription(Odometry, t.odometry, self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Float64MultiArray, t.path, self.on_path, 10)
        self.create_subscription(Float32, t.speed_target, lambda m: setattr(self, "speed_target", float(m.data)), 10)
        self.create_subscription(Float32, t.steer_feedback, lambda m: setattr(self, "steer_fb", float(m.data)), 10)
        self.create_subscription(Float32, t.wheel_speed, lambda m: setattr(self, "wheel_speed", float(m.data)), 10)
        self.create_subscription(String, t.mission_status, self.on_mission, 10)
        self.steer_pub = self.create_publisher(Float32, t.steer_cmd, 10)
        self.speed_pub = self.create_publisher(Float32, t.speed_cmd, 10)
        self.status_pub = self.create_publisher(String, t.control_status, 10)
        self.dt = 1.0 / max(float(self.cfg.control.rate_hz), 1.0)
        # Own timer: YOLO / LiDAR / planning live in other processes and cannot block this loop.
        self.create_timer(self.dt, self.tick)

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_odom(self, msg):
        self.odom.update(msg)
        self.odom_wall = self.now()

    def on_path(self, msg):
        data = list(msg.data)
        self.ref = ReferenceLine(data[0], data[1], data[2])
        self.path = OffsetPath.from_list(data[3:])

    def on_mission(self, msg):
        try:
            self.mission = json.loads(msg.data)
        except ValueError:
            pass

    def tick(self):
        cfg = self.cfg
        now = self.now()
        odom_ok = self.odom.valid and now - self.odom_wall <= cfg.control.odom_timeout
        active = self.ref is not None and self.speed_target > 0.0
        out = None
        if active and odom_ok:
            x_f, y_f, yaw = self.odom.front_axle_at(now)
            out = self.stanley.step(self.ref, self.path, x_f, y_f, yaw, self.odom.v, self.dt)
            delta = out.delta
        else:
            # Pre-start: wheels straight. Localisation lost mid-run: hold the last angle
            # (the path is spatial and stays valid, but there is no pose to steer by).
            delta = self.stanley.delta_prev if active else 0.0
            if not active:
                self.stanley.reset()

        track_err = abs(delta - self.steer_fb) if math.isfinite(self.steer_fb) else 0.0
        if active and track_err > math.radians(cfg.control.steer_track_err_max_deg):
            self.track_err_since = self.track_err_since or now
        else:
            self.track_err_since = None
        steer_fault = (self.track_err_since is not None
                       and now - self.track_err_since >= cfg.control.steer_track_err_time)
        fault = steer_fault or (active and not odom_ok)

        speed = self.speed_target
        if fault and speed > 0.0:
            speed = min(speed, cfg.speed.fault)   # slow down; never stop
        angle_out, speed_out = to_cmd_units(delta, speed, cfg)
        self.steer_pub.publish(Float32(data=float(angle_out)))
        self.speed_pub.publish(Float32(data=float(speed_out)))
        self.status_pub.publish(String(data=json.dumps({
            "fault": bool(fault), "steer_tracking_fault": bool(steer_fault),
            "odom_timeout": bool(active and not odom_ok), "track_err": track_err})))

        out_rad = wheel_to_output(delta, cfg)
        m = self.mission
        self.log.write({
            "t": now, "rtk_ok": m.get("rtk_ok"), "x": self.odom.x, "y": self.odom.y,
            "yaw": self.odom.yaw, "v": self.odom.v,
            "s": out.s if out else math.nan, "d": out.d if out else math.nan,
            "state": m.get("state"), "d_target_raw": m.get("cur_d_raw"),
            "d_target_filtered": m.get("cur_d_target"), "plan_L": m.get("plan_L"),
            "d_path": out.d_path if out else math.nan, "e_y": out.e_y if out else math.nan,
            "theta_e": out.theta_e if out else math.nan,
            "delta_wheel": delta, "delta_output": out_rad,
            "steer_cmd_published": angle_out, "speed_cmd_published": speed_out,
            "steer_fb_wheel": self.steer_fb, "track_err": track_err,
            "v_cmd": speed, "wheel_speed": self.wheel_speed,
            "clearance_circle": m.get("clearance_circle"), "clearance_rect": m.get("clearance_rect"),
            "v_allow": m.get("v_allow"), "plan_level": m.get("plan_level"),
            "fault_steer": steer_fault, "fault_odom": active and not odom_ok,
            "flags": "|".join(m.get("flags", [])), "faults": "|".join(m.get("faults", [])),
        })

    def destroy_node(self):
        self.log.close()
        super().destroy_node()


def main(args=None):
    run_node(ControlNode)


if __name__ == "__main__":
    main()
