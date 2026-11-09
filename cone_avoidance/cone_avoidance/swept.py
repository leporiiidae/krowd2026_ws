"""Swept-path and steering-demand evaluation of an offset path.

The front axle is assumed to follow the path. The body yaw is NOT taken equal
to the path tangent: the rear axle is dragged behind the front axle (tractrix),

    d(psi)/d(sigma) = sin(theta - psi) / wheelbase

with sigma the front-axle arc length and theta the path tangent. In a lane
change the body yaw lags the tangent and the rear axle cuts inside the front
axle's track. Since the rearmost point of this vehicle is the rear tyre, using
the tangent as body yaw would misplace exactly the part that hits the cone.

The same integration yields the wheel angle the path demands, delta = theta -
psi, so curvature and steering-rate limits are checked on the real demand
instead of the rest-to-rest closed forms.

All coordinates here are Frenet (s, d) of the straight reference line, which is
a rigid transform of the global frame, so distances are exact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .geometry import OffsetPath


@dataclass
class Trace:
    s: np.ndarray        # front axle s
    d: np.ndarray        # front axle d
    theta: np.ndarray    # path tangent (relative to reference heading)
    psi: np.ndarray      # body yaw (relative to reference heading)
    delta: np.ndarray    # demanded wheel angle
    ddelta_ds: np.ndarray


@dataclass
class Evaluation:
    clearance_circle: float      # 3-circle approximation (conservative, used for acceptance)
    clearance_rect: float        # exact rectangle (logged for comparison)
    worst_s: float               # front-axle s where clearance_circle is minimal
    worst_cone: int              # index of the limiting cone (-1 if no cones)
    max_delta: float
    max_ddelta_ds: float         # rad per metre of travel
    body_left: float = -math.inf   # largest d reached by any body corner
    body_right: float = math.inf   # smallest d reached by any body corner
    trace: Optional[Trace] = None

    def v_feasible(self, steer_rate: float) -> float:
        """Highest speed at which the demanded steering rate stays within ``steer_rate``."""
        if self.max_ddelta_ds <= 1e-9:
            return math.inf
        return steer_rate / self.max_ddelta_ds


def trace_path(path: OffsetPath, s_start: float, s_end: float, psi0: float,
               wheelbase: float, step: float) -> Trace:
    n = max(2, int(math.ceil((s_end - s_start) / step)) + 1)
    # Half-step grid so RK4 needs no extra path evaluations inside the loop.
    fine = np.linspace(s_start, s_end, 2 * n - 1)
    d, d1, _ = path.eval_array(fine)
    theta_f = np.arctan(d1)
    stretch_f = np.sqrt(1.0 + d1 * d1) / wheelbase
    h = (s_end - s_start) / (n - 1)

    psi = np.empty(n)
    psi[0] = psi0
    p = psi0
    sin = math.sin
    th = theta_f.tolist()
    st = stretch_f.tolist()
    for i in range(n - 1):
        j = 2 * i
        k1 = sin(th[j] - p) * st[j]
        k2 = sin(th[j + 1] - (p + 0.5 * h * k1)) * st[j + 1]
        k3 = sin(th[j + 1] - (p + 0.5 * h * k2)) * st[j + 1]
        k4 = sin(th[j + 2] - (p + h * k3)) * st[j + 2]
        p += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        psi[i + 1] = p

    s = fine[::2]
    theta = theta_f[::2]
    delta = theta - psi
    ddelta = np.gradient(delta, h) if n > 2 else np.zeros(n)
    return Trace(s, d[::2], theta, psi, delta, ddelta)


def body_circles(cfg):
    """Three covering circles: centres measured forward from the REAR axle."""
    v = cfg.vehicle
    rear = -v.rear_overhang
    front = v.wheelbase + v.front_overhang
    body_len = front - rear
    radius = math.hypot(body_len / 6.0, cfg.collision_width / 2.0)
    centers = np.array([rear + body_len * f for f in (1.0 / 6.0, 0.5, 5.0 / 6.0)])
    return centers, radius


def rect_distance(px, py, rear_x, rear_y, psi, cfg):
    """Distance from points to the body rectangle (0 inside). Arrays broadcast."""
    v = cfg.vehicle
    c, s = np.cos(psi), np.sin(psi)
    dx, dy = px - rear_x, py - rear_y
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    x_lo, x_hi = -v.rear_overhang, v.wheelbase + v.front_overhang
    half_w = cfg.collision_width / 2.0
    ox = np.maximum(np.maximum(x_lo - lx, lx - x_hi), 0.0)
    oy = np.maximum(np.abs(ly) - half_w, 0.0)
    return np.hypot(ox, oy)


def clearance_at_pose(front_s, front_d, psi, cones_sd: np.ndarray, cfg):
    """(circle, rect) clearance for explicit poses; used by the simulator too."""
    wb = cfg.vehicle.wheelbase
    front_s = np.atleast_1d(np.asarray(front_s, dtype=float))
    front_d = np.atleast_1d(np.asarray(front_d, dtype=float))
    psi = np.atleast_1d(np.asarray(psi, dtype=float))
    rear_s = front_s - wb * np.cos(psi)
    rear_d = front_d - wb * np.sin(psi)
    centers, radius = body_circles(cfg)
    cs = cones_sd[:, 0][None, None, :]
    cd = cones_sd[:, 1][None, None, :]
    cx = rear_s[:, None, None] + centers[None, :, None] * np.cos(psi)[:, None, None]
    cy = rear_d[:, None, None] + centers[None, :, None] * np.sin(psi)[:, None, None]
    circle = np.hypot(cx - cs, cy - cd).min(axis=1) - radius - cfg.cone_radius   # (pose, cone)
    rect = rect_distance(cones_sd[:, 0][None, :], cones_sd[:, 1][None, :],
                         rear_s[:, None], rear_d[:, None], psi[:, None], cfg) - cfg.cone_radius
    return circle, rect


def body_edges(front_d, psi, cfg):
    """Lateral extent (min d, max d) of the collision rectangle at each pose."""
    v = cfg.vehicle
    wb = v.wheelbase
    rear_d = np.asarray(front_d) - wb * np.sin(psi)
    xs = np.array([-v.rear_overhang, wb + v.front_overhang])
    ys = np.array([-1.0, 1.0]) * cfg.collision_width / 2.0
    s_, c_ = np.sin(psi)[:, None], np.cos(psi)[:, None]
    corners = rear_d[:, None] + np.concatenate([xs[None, :] * s_ + y * c_ for y in ys], axis=1)
    return corners.min(axis=1), corners.max(axis=1)


def _grid(s_now: float, end_s: float, cones_sd, cfg):
    s_end = max(end_s, s_now) + 1.0
    ahead = cones_sd[cones_sd[:, 0] > s_now - cfg.rear_axle_to_front] if len(cones_sd) else cones_sd
    if len(ahead):
        s_end = max(s_end, float(ahead[:, 0].max()) + cfg.front_axle_to_rear + 1.0)
    n = max(2, int(math.ceil((s_end - s_now) / cfg.planning.sample_step)) + 1)
    return np.linspace(s_now, s_end, 2 * n - 1), n, (s_end - s_now) / (n - 1)   # half-step grid


def evaluate_many(paths, s_now: float, psi_now: float, cones_sd, cfg, keep_trace: bool = False):
    """Score several candidate paths at once (same start pose, same cones)."""
    paths = list(paths)
    if not paths:
        return []
    cones_sd = np.asarray(cones_sd, dtype=float).reshape(-1, 2)
    fine, n, h = _grid(s_now, max(p.end_s for p in paths), cones_sd, cfg)
    evals = [p.eval_array(fine) for p in paths]
    return _score(fine, n, h, np.stack([e[0] for e in evals]), np.stack([e[1] for e in evals]),
                  s_now, psi_now, cones_sd, cfg, keep_trace)


def evaluate_branches(base: OffsetPath, segments, s_now: float, psi_now: float, cones_sd, cfg,
                      keep_trace: bool = False, with_rect: bool = True):
    """Score candidates of the form "``base`` up to the segment start, then the segment".

    Every planner candidate has this shape, so the shared part is evaluated once and the
    new segments as one (candidates x samples) polynomial — no per-candidate Python.
    """
    segments = list(segments)
    if not segments:
        return []
    cones_sd = np.asarray(cones_sd, dtype=float).reshape(-1, 2)
    fine, n, h = _grid(s_now, max(seg.s1 for seg in segments), cones_sd, cfg)
    base_d, base_d1, _ = base.eval_array(fine)
    s0 = np.array([seg.s0 for seg in segments])[:, None]
    length = np.array([seg.length for seg in segments])[:, None]
    a = np.array([seg.coeffs for seg in segments])                      # (C, 6)
    u = np.clip((fine[None, :] - s0) / length, 0.0, 1.0)
    d_seg = a[:, [0]] + u * (a[:, [1]] + u * (a[:, [2]] + u * (a[:, [3]] + u * (a[:, [4]] + u * a[:, [5]]))))
    d1_seg = (a[:, [1]] + u * (2 * a[:, [2]] + u * (3 * a[:, [3]] + u * (4 * a[:, [4]] + u * 5 * a[:, [5]])))) / length
    after = fine[None, :] >= s0            # beyond s1 the clipped polynomial holds the end offset
    d_f = np.where(after, d_seg, base_d[None, :])
    d1_f = np.where(after, d1_seg, base_d1[None, :])
    return _score(fine, n, h, d_f, d1_f, s_now, psi_now, cones_sd, cfg, keep_trace, with_rect)


def _score(fine, n, h, d_f, d1_f, s_now, psi_now, cones_sd, cfg, keep_trace, with_rect=True):
    """The tractrix integration is sequential along s but independent across candidates,
    so RK4 runs vectorised over the candidate axis. The planner scores a few hundred
    candidates inside one 10 Hz tick; one by one in Python that took ~190 ms."""
    wb = cfg.vehicle.wheelbase
    theta_f = np.arctan(d1_f)
    stretch_f = np.sqrt(1.0 + d1_f * d1_f) / wb
    count = d_f.shape[0]

    psi = np.empty((count, n))
    p = np.full(count, float(psi_now))
    psi[:, 0] = p
    for i in range(n - 1):
        j = 2 * i
        k1 = np.sin(theta_f[:, j] - p) * stretch_f[:, j]
        k2 = np.sin(theta_f[:, j + 1] - (p + 0.5 * h * k1)) * stretch_f[:, j + 1]
        k3 = np.sin(theta_f[:, j + 1] - (p + 0.5 * h * k2)) * stretch_f[:, j + 1]
        k4 = np.sin(theta_f[:, j + 2] - (p + h * k3)) * stretch_f[:, j + 2]
        p = p + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        psi[:, i + 1] = p

    s = fine[::2]
    d = d_f[:, ::2]
    theta = theta_f[:, ::2]
    delta = theta - psi
    ddelta = np.gradient(delta, h, axis=1) if n > 2 else np.zeros_like(delta)
    max_delta = np.abs(delta).max(axis=1)
    max_rate = np.abs(ddelta).max(axis=1)

    # Body geometry for every candidate and pose.
    v = cfg.vehicle
    cos_p, sin_p = np.cos(psi), np.sin(psi)
    rear_s = s[None, :] - wb * cos_p
    rear_d = d - wb * sin_p
    half_w = cfg.collision_width / 2.0
    xs = np.array([-v.rear_overhang, wb + v.front_overhang])
    corner_d = np.stack([rear_d + x * sin_p + y * cos_p for x in xs for y in (-half_w, half_w)])
    body_right = corner_d.min(axis=(0, 2))
    body_left = corner_d.max(axis=(0, 2))

    out = []
    if len(cones_sd):
        centers, radius = body_circles(cfg)
        cs, cd = cones_sd[:, 0], cones_sd[:, 1]
        circle = np.full((count, n, len(cs)), np.inf)
        for c0 in centers:
            cx = rear_s + c0 * cos_p
            cy = rear_d + c0 * sin_p
            dist = np.hypot(cx[:, :, None] - cs, cy[:, :, None] - cd)
            circle = np.minimum(circle, dist)
        circle -= radius + cfg.cone_radius
    if len(cones_sd) and with_rect:
        dx = cs - rear_s[:, :, None]
        dy = cd - rear_d[:, :, None]
        lx = cos_p[:, :, None] * dx + sin_p[:, :, None] * dy
        ly = -sin_p[:, :, None] * dx + cos_p[:, :, None] * dy
        ox = np.maximum(np.maximum(xs[0] - lx, lx - xs[1]), 0.0)
        oy = np.maximum(np.abs(ly) - half_w, 0.0)
        rect = np.hypot(ox, oy) - cfg.cone_radius
    for c in range(count):
        if len(cones_sd):
            flat = int(np.argmin(circle[c]))
            pose_i, cone_i = np.unravel_index(flat, circle[c].shape)
            clr_c = float(circle[c, pose_i, cone_i])
            clr_r = float(rect[c].min()) if with_rect else math.nan
            worst_s, worst_cone = float(s[pose_i]), int(cone_i)
        else:
            clr_c = clr_r = math.inf
            worst_s, worst_cone = s_now, -1
        trace = (Trace(s, d[c], theta[c], psi[c], delta[c], ddelta[c]) if keep_trace else None)
        out.append(Evaluation(clearance_circle=clr_c, clearance_rect=clr_r, worst_s=worst_s,
                              worst_cone=worst_cone, max_delta=float(max_delta[c]),
                              max_ddelta_ds=float(max_rate[c]), body_left=float(body_left[c]),
                              body_right=float(body_right[c]), trace=trace))
    return out


def evaluate(path: OffsetPath, s_now: float, psi_now: float, cones_sd, cfg,
             s_end: Optional[float] = None, keep_trace: bool = False) -> Evaluation:
    """Trace ``path`` from the current front-axle station and score it.

    ``cones_sd``: (N, 2) cone centres in Frenet. The trace runs until the whole
    body has passed the last cone and the last segment has ended.
    """
    cones_sd = np.asarray(cones_sd, dtype=float).reshape(-1, 2)
    if s_end is None:
        s_end = max(path.end_s, s_now) + 1.0
        ahead = cones_sd[cones_sd[:, 0] > s_now - cfg.rear_axle_to_front] if len(cones_sd) else cones_sd
        if len(ahead):
            s_end = max(s_end, float(ahead[:, 0].max()) + cfg.front_axle_to_rear + 1.0)
    trace = trace_path(path, s_now, s_end, psi_now, cfg.vehicle.wheelbase, cfg.planning.sample_step)

    if len(cones_sd):
        circle, rect = clearance_at_pose(trace.s, trace.d, trace.psi, cones_sd, cfg)
        flat = int(np.argmin(circle))
        pose_i, cone_i = np.unravel_index(flat, circle.shape)
        clr_c, clr_r = float(circle[pose_i, cone_i]), float(rect.min())
        worst_s, worst_cone = float(trace.s[pose_i]), int(cone_i)
    else:
        clr_c = clr_r = math.inf
        worst_s, worst_cone = s_now, -1

    right, left = body_edges(trace.d, trace.psi, cfg)
    return Evaluation(
        body_left=float(left.max()),
        body_right=float(right.min()),
        clearance_circle=clr_c,
        clearance_rect=clr_r,
        worst_s=worst_s,
        worst_cone=worst_cone,
        max_delta=float(np.abs(trace.delta).max()),
        max_ddelta_ds=float(np.abs(trace.ddelta_ds).max()),
        trace=trace if keep_trace else None,
    )
