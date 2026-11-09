"""Stanley control node (50 Hz) — also the per-cycle logger."""

from __future__ import annotations

import json
import math

from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Float32, Float64MultiArray, String

from .can_codec import encode_steer, to_cmd_units
from .csv_logger import CsvLogger
from .geometry import OffsetPath, ReferenceLine
from .ros_common import run_node, OdomState, declare_config
from .stanley import StanleyController


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

        ratio = cfg.steering.output_ratio
        _, out_rad, raw = encode_steer(delta, ratio, cfg.steering.output_limit_rad, cfg.steering.angle_scale)
        m = self.mission
        self.log.write({
            "t": now, "rtk_ok": m.get("rtk_ok"), "x": self.odom.x, "y": self.odom.y,
            "yaw": self.odom.yaw, "v": self.odom.v,
            "s": out.s if out else math.nan, "d": out.d if out else math.nan,
            "state": m.get("state"), "d_target_raw": m.get("cur_d_raw"),
            "d_target_filtered": m.get("cur_d_target"), "plan_L": m.get("plan_L"),
            "d_path": out.d_path if out else math.nan, "e_y": out.e_y if out else math.nan,
            "theta_e": out.theta_e if out else math.nan,
            "delta_wheel": delta, "delta_output": out_rad, "can_raw": raw,
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
