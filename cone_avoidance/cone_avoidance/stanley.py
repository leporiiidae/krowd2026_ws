"""Stanley lateral controller on the front axle, with latency compensation."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import OffsetPath, ReferenceLine, wrap


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
