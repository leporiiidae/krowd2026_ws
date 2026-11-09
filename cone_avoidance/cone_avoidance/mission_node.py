"""Cone mission node with geometry, planning and swept-path evaluation."""

from __future__ import annotations

"""Coordinate conventions, Frenet transform and quintic offset paths.

Sign conventions (spec section 3) — every module relies on these:
  * global frame: local ENU, yaw counter-clockwise positive
  * Frenet: ``s`` along the straight reference line, ``d`` lateral, LEFT positive
  * wheel steering angle delta: LEFT turn positive
  * control reference point: FRONT axle centre
"""


import math
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np


def wrap(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class ReferenceLine:
    """Straight reference line: curvature is zero, so Frenet is one rigid transform."""

    origin_x: float
    origin_y: float
    heading: float

    def to_frenet(self, x: float, y: float) -> Tuple[float, float]:
        dx, dy = x - self.origin_x, y - self.origin_y
        c, s = math.cos(self.heading), math.sin(self.heading)
        return (c * dx + s * dy, -s * dx + c * dy)

    def to_frenet_array(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        c, s = math.cos(self.heading), math.sin(self.heading)
        dx, dy = xy[:, 0] - self.origin_x, xy[:, 1] - self.origin_y
        return np.column_stack((c * dx + s * dy, -s * dx + c * dy))

    def to_global(self, s: float, d: float) -> Tuple[float, float]:
        c, sn = math.cos(self.heading), math.sin(self.heading)
        return (self.origin_x + c * s - sn * d, self.origin_y + sn * s + c * d)

    def relative_yaw(self, yaw: float) -> float:
        return wrap(yaw - self.heading)


def quintic_coefficients(
    d0: float, dd0: float, ddd0: float,
    d1: float, dd1: float, ddd1: float,
    length: float,
) -> Tuple[float, ...]:
    """General quintic in u = (s - s0) / L with full boundary conditions.

    Derivatives are with respect to s (d' = dd/ds), as in spec 7.2.
    """
    a0 = d0
    a1 = dd0 * length
    a2 = ddd0 * length * length / 2.0
    A = d1 - (a0 + a1 + a2)
    B = dd1 * length - (a1 + 2.0 * a2)
    C = ddd1 * length * length - 2.0 * a2
    a3 = 10.0 * A - 4.0 * B + 0.5 * C
    a4 = -15.0 * A + 7.0 * B - C
    a5 = 6.0 * A - 3.0 * B + 0.5 * C
    return (a0, a1, a2, a3, a4, a5)


@dataclass(frozen=True)
class QuinticSegment:
    s0: float
    s1: float
    coeffs: Tuple[float, ...]

    @classmethod
    def connect(cls, s0, start: Sequence[float], s1, end: Sequence[float]) -> "QuinticSegment":
        length = s1 - s0
        if length <= 1e-6:
            raise ValueError("segment length must be positive")
        return cls(s0, s1, quintic_coefficients(*start, *end, length))

    @property
    def length(self) -> float:
        return self.s1 - self.s0

    def eval(self, s):
        """Return (d, d', d'') at s (scalar or array), s clamped into the segment."""
        a0, a1, a2, a3, a4, a5 = self.coeffs
        L = self.length
        u = np.clip((np.asarray(s, dtype=float) - self.s0) / L, 0.0, 1.0)
        d = a0 + u * (a1 + u * (a2 + u * (a3 + u * (a4 + u * a5))))
        d1 = (a1 + u * (2 * a2 + u * (3 * a3 + u * (4 * a4 + u * 5 * a5)))) / L
        d2 = (2 * a2 + u * (6 * a3 + u * (12 * a4 + u * 20 * a5))) / (L * L)
        return d, d1, d2

    @property
    def end_offset(self) -> float:
        return float(sum(self.coeffs))


@dataclass
class OffsetPath:
    """Piecewise lateral offset d(s).

    Before the first segment the offset is ``base_offset``; after a segment ends
    the offset holds that segment's end value until the next segment starts.
    Segments must end with zero slope/curvature for the hold to stay C2, which
    every planner-made segment does.
    """

    segments: List[QuinticSegment] = field(default_factory=list)
    base_offset: float = 0.0

    def copy(self) -> "OffsetPath":
        return OffsetPath(list(self.segments), self.base_offset)

    def eval(self, s: float) -> Tuple[float, float, float]:
        hold = self.base_offset
        for seg in self.segments:
            if s < seg.s0:
                break
            if s <= seg.s1:
                d, d1, d2 = seg.eval(s)
                return float(d), float(d1), float(d2)
            hold = seg.end_offset
        return hold, 0.0, 0.0

    def eval_array(self, s: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.asarray(s, dtype=float)
        d = np.full(s.shape, self.base_offset)
        d1 = np.zeros(s.shape)
        d2 = np.zeros(s.shape)
        for seg in self.segments:
            inside = (s >= seg.s0) & (s <= seg.s1)
            after = s > seg.s1
            if inside.any():
                vd, vd1, vd2 = seg.eval(s[inside])
                d[inside], d1[inside], d2[inside] = vd, vd1, vd2
            d[after] = seg.end_offset
            d1[after] = 0.0
            d2[after] = 0.0
        return d, d1, d2

    def truncated_before(self, s: float) -> "OffsetPath":
        """Keep only segments that start before ``s`` (used when re-planning ahead)."""
        return OffsetPath([seg for seg in self.segments if seg.s0 < s], self.base_offset)

    def replace_from(self, s: float, new_segments: Sequence[QuinticSegment]) -> "OffsetPath":
        """Cut the path at ``s`` and continue with ``new_segments``.

        A segment in progress at ``s`` is shortened so its evaluated values up to
        ``s`` are unchanged; the first new segment must start at ``s`` with the
        path state there for the result to stay C2.
        """
        kept: List[QuinticSegment] = []
        for seg in self.segments:
            if seg.s1 <= s:
                kept.append(seg)
            elif seg.s0 < s:
                kept.append(_head(seg, s))
        return OffsetPath(kept + list(new_segments), self.base_offset)

    @property
    def end_s(self) -> float:
        return self.segments[-1].s1 if self.segments else -math.inf

    # -- serialisation for the ROS path topic ---------------------------------
    def to_list(self) -> List[float]:
        out = [self.base_offset, float(len(self.segments))]
        for seg in self.segments:
            out.extend([seg.s0, seg.s1, *seg.coeffs])
        return out

    @classmethod
    def from_list(cls, data: Sequence[float]) -> "OffsetPath":
        base, count = float(data[0]), int(round(data[1]))
        segments = []
        for i in range(count):
            row = data[2 + 8 * i: 10 + 8 * i]
            segments.append(QuinticSegment(float(row[0]), float(row[1]), tuple(map(float, row[2:8]))))
        return cls(segments, base)


def _head(seg: QuinticSegment, s_cut: float) -> QuinticSegment:
    """Exact re-parameterisation of ``seg`` restricted to [s0, s_cut]."""
    ratio = (s_cut - seg.s0) / seg.length
    coeffs = tuple(c * ratio ** k for k, c in enumerate(seg.coeffs))
    return QuinticSegment(seg.s0, s_cut, coeffs)


# Analytic properties of the rest-to-rest quintic (spec 7.2), used for design.
QUINTIC_MAX_D1 = 1.875
QUINTIC_MAX_D2 = 10.0 / math.sqrt(3.0)
QUINTIC_MAX_D3 = 60.0
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


import math
from dataclasses import dataclass
from typing import Optional

import numpy as np



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
"""Offset-path planning: target offsets, maneuver search, re-planning.

Speed is decided here, not given: every candidate path is traced with the real
body kinematics (``swept.evaluate``), which yields both the clearance against
every cone and the steering rate the path demands per metre. The demanded rate
fixes the highest speed the stepper can follow, so "how fast may we go" and
"does the body fit" are answered by the same evaluation. The closed-form speed
of spec 7.4 is still computed (with the body length removed from the gap) and
logged as a reference.

Failure ladder (spec 10) — a plan is always returned, never "stop":
  level 0  nominal candidate accepted
  level 1  accepted after growing the offset toward the lane limit
  level 2  no candidate has positive clearance: widest one is used, warning
  level 3  nothing the steering can follow: caller keeps the existing path
"""


import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .perception_node import ConePair


@dataclass
class VehicleState:
    s: float          # front axle
    d: float
    psi: float        # yaw relative to the reference heading
    v: float


@dataclass
class TargetOffset:
    d: float          # clipped target
    d_raw: float      # before lane clipping
    side: int         # +1 pass on the LEFT of the group, -1 on the right
    clipped: bool
    lo: float = -math.inf   # offset range the clearance ladder may use
    hi: float = math.inf


@dataclass
class PlanResult:
    path: OffsetPath
    segment: Optional[QuinticSegment]
    evaluation: Optional[Evaluation]
    v_allow: float
    level: int
    d_end: float = 0.0
    v_allow_analytic: float = math.nan
    note: str = ""


class _LazyPath:
    """Defers splicing the winning segment into the path until a candidate is chosen."""

    def __init__(self, base: OffsetPath, seg: QuinticSegment):
        self.base, self.seg = base, seg

    def build(self) -> OffsetPath:
        return self.base.replace_from(self.seg.s0, [self.seg])


@dataclass
class _Candidate:
    path: OffsetPath
    segment: QuinticSegment
    ev: Evaluation
    v_feas: float
    steer_ok: bool
    d_end: float
    lane_ok: bool = True


class Planner:
    def __init__(self, cfg):
        self.cfg = cfg

    # ------------------------------------------------------------------ 7.1
    def limits(self, index: int = 1, lane_c: float = 0.0):
        """Allowed front-axle offset range (lo, hi) while handling obstacle group ``index``.

        ``lane_c`` is the lane centre in reference coordinates. The lane's LEFT boundary
        is the centre line, the RIGHT one the solid line. The solid line is never
        crossed; the centre line may be crossed by ``road.center_cross_extra`` for the
        groups listed in ``road.center_cross_groups`` (no penalty there by the rules).
        """
        cfg = self.cfg
        lo, hi = lane_c - cfg.d_limit, lane_c + cfg.d_limit
        if int(index) in [int(i) for i in cfg.road.center_cross_groups]:
            hi += cfg.road.center_cross_extra
        return lo, hi

    def lane_center_from(self, pair: ConePair, stands: str) -> float:
        """Lane centre implied by a group that stands flush against a lane line.

        The rules put group 1 against the centre line and group 2 against the solid
        line, so the OUTER edge of the row IS that line. This anchors the lane on the
        obstacles themselves instead of on a reference line fixed 30+ m earlier, whose
        1 deg of heading error is already 0.6 m of lateral error out here.
        """
        cfg = self.cfg
        if stands == "left":
            return float(pair.d.max()) + cfg.cone_radius - cfg.road.width / 2.0
        return float(pair.d.min()) - cfg.cone_radius + cfg.road.width / 2.0

    def blocks(self, pair: ConePair, lane_c: float = 0.0) -> bool:
        """True for an obstacle group standing IN the lane, false for roadside clutter.

        Deliberately not "does the lane-centre path need to move": that flips on a few
        centimetres of reference error, and a skipped group shifts the group numbering.
        A real group reaches over a metre into the lane; vegetation stays at the line.
        """
        cfg = self.cfg
        inner_edge = float(np.abs(pair.d - lane_c).min()) - cfg.cone_radius
        return inner_edge < cfg.road.width / 2.0 - cfg.mission.min_intrusion

    def target_offset(self, pair: ConePair, index: int = 1, lane_c: float = 0.0,
                      stands: Optional[str] = None) -> TargetOffset:
        """``stands``: "left"/"right" when the rules say where the group is (then the pass
        side is the other one, whatever the coordinates look like); None = decide here."""
        cfg = self.cfg
        reach = cfg.cone_radius + cfg.collision_width / 2.0 + cfg.planning.lateral_margin
        # The offset is a lower bound on how far to move AWAY; never steer toward the group.
        left = max(float(pair.d.max()) + reach, lane_c)     # pass on the left of every cone
        right = min(float(pair.d.min()) - reach, lane_c)    # pass on the right
        lo, hi = self.limits(index, lane_c)

        if stands == "left":
            d_raw, side = right, -1
        elif stands == "right":
            d_raw, side = left, +1
        else:
            # Least lane violation first, then the smaller move.
            key_left = (max(0.0, left - hi), abs(left - lane_c))
            key_right = (max(0.0, lo - right), abs(right - lane_c))
            d_raw, side = (left, +1) if key_left <= key_right else (right, -1)
        if cfg.road.lane_hard_constraint:
            d = min(max(d_raw, lo), hi)
        else:
            d = d_raw
            lo, hi = min(lo, d_raw) - 0.5, max(hi, d_raw) + 0.5
        return TargetOffset(d, d_raw, side, abs(d - d_raw) > 1e-9, lo, hi)

    # ------------------------------------------------------------------ 7.3
    def analytic_length(self, swing: float, v: float) -> float:
        cfg = self.cfg
        swing = abs(swing)
        if swing < 1e-6:
            return cfg.planning.plan_length_min
        l_curv = math.sqrt(QUINTIC_MAX_D2 * swing / cfg.kappa_limit)
        l_rate = (QUINTIC_MAX_D3 * swing * max(v, 0.0) * cfg.vehicle.wheelbase
                  / cfg.plan_steer_rate) ** (1.0 / 3.0)
        return max(l_curv, l_rate, cfg.planning.plan_length_min)

    def analytic_speed(self, length: float, swing: float) -> float:
        if length <= 0.0:
            return 0.0
        if abs(swing) < 1e-6:
            return math.inf
        return (self.cfg.plan_steer_rate * length ** 3
                / (QUINTIC_MAX_D3 * abs(swing) * self.cfg.vehicle.wheelbase))

    def body_free_length(self, a: ConePair, b: ConePair) -> float:
        """Stretch where the body is abeam neither pair (spec 7.4, body removed)."""
        cfg = self.cfg
        return (b.s_min - a.s_max) - (
            cfg.front_axle_to_rear + cfg.vehicle.front_overhang + 2.0 * cfg.cone_radius)

    # ------------------------------------------------------------ candidates
    def _v_target(self) -> float:
        return min(self.cfg.speed.target, self.cfg.speed.max_clamp)

    def _candidates(self, base, state: VehicleState, specs, cones_sd, limits=None) -> List[_Candidate]:
        """Build and score candidates ``specs`` = [(base_path, s0, s1, d_end), ...] in one batch.

        ``limits``: allowed front-axle range (lo, hi); every body corner must stay within
        it widened by half the collision width (= the lane lines).
        """
        built = []
        for base_path, s0, s1, d_end in specs:
            if s1 - s0 < 0.3:
                continue
            seg = QuinticSegment.connect(s0, base_path.eval(s0), s1, (d_end, 0.0, 0.0))
            built.append((base_path, seg, d_end))
        # Before its segment starts every candidate IS ``base``, so score them as branches.
        evs = evaluate_branches(base, [b[1] for b in built], state.s, state.psi, cones_sd, self.cfg,
                                with_rect=False)   # exact-rectangle value: winner only (it is a log field)
        half = self.cfg.collision_width / 2.0
        tol = self.cfg.road.lane_tolerance
        check_lane = limits is not None and self.cfg.road.lane_hard_constraint
        out = []
        for (base_path, seg, d_end), ev in zip(built, evs):
            path = _LazyPath(base_path, seg)
            lane_ok = (not check_lane) or (ev.body_left <= limits[1] + half + tol
                                           and ev.body_right >= limits[0] - half - tol)
            out.append(_Candidate(path, seg, ev, ev.v_feasible(self.cfg.plan_steer_rate),
                                  ev.max_delta <= self.cfg.plan_max_steer, d_end, lane_ok))
        self._scoring = (state, cones_sd)
        return out

    def _candidate(self, base: OffsetPath, state: VehicleState, s0: float, s1: float,
                   d_end: float, cones_sd, limits=None) -> Optional[_Candidate]:
        found = self._candidates(base, state, [(base, s0, s1, d_end)], cones_sd, limits)
        return found[0] if found else None

    def _pick(self, cands: Sequence[_Candidate]) -> Optional[_Candidate]:
        """Best nominal candidate: fastest speed bucket, then clearance bucket, then gentlest."""
        cfg = self.cfg
        v_target = self._v_target()
        step = max(cfg.failure.fail_speed_step, 1e-3)
        ok = [c for c in cands
              if c.lane_ok and c.steer_ok and c.v_feas >= cfg.failure.fail_min_speed
              and c.ev.clearance_circle >= cfg.planning.min_clearance]
        if not ok:
            return None
        # Clearance differences below the bucket are estimate noise, not merit; among
        # such ties take the gentlest steering so tracking error stays small.
        bucket = max(cfg.planning.clearance_bucket, 1e-3)
        return max(ok, key=lambda c: (math.floor(min(c.v_feas, v_target) / step + 1e-9),
                                      math.floor(c.ev.clearance_circle / bucket),
                                      -c.ev.max_ddelta_ds))

    def _pick_by_rung(self, cands: Sequence[_Candidate], ladder: Sequence[float], base: OffsetPath,
                      note: str, analytic: float = math.nan) -> PlanResult:
        """All rungs were scored in one batch (constant planning time); choose as before:
        the first rung — smallest extra offset — that has an acceptable candidate."""
        for i, d_end in enumerate(ladder):
            best = self._pick([c for c in cands if c.d_end == d_end])
            if best:
                return self._result(best, 0 if i == 0 else 1, base, note, analytic)
        cand, level = self._fallback(cands)
        return self._result(cand, level, base, note, analytic)

    def _fallback(self, cands: Sequence[_Candidate]) -> Tuple[Optional[_Candidate], int]:
        followable = [c for c in cands
                      if c.steer_ok and c.v_feas >= self.cfg.failure.fail_min_speed]
        if followable:
            # Lane lines stay hard even here; only the cone clearance is given up.
            in_lane = [c for c in followable if c.lane_ok] or followable
            return max(in_lane, key=lambda c: c.ev.clearance_circle), 2
        if cands:
            return max(cands, key=lambda c: (c.v_feas, c.ev.clearance_circle)), 3
        return None, 3

    def _result(self, cand: Optional[_Candidate], level: int, base: OffsetPath, note: str,
                analytic: float = math.nan) -> PlanResult:
        if cand is None:
            return PlanResult(base, None, None, self.cfg.failure.fail_min_speed, 3, 0.0,
                              analytic, note + " no-candidate")
        v_allow = max(min(cand.v_feas, self._v_target()), self.cfg.failure.fail_min_speed)
        path = cand.path.build() if isinstance(cand.path, _LazyPath) else cand.path
        if math.isnan(cand.ev.clearance_rect):
            state, cones_sd = self._scoring
            cand.ev.clearance_rect = evaluate(path, state.s, state.psi, cones_sd, self.cfg).clearance_rect
        return PlanResult(path, cand.segment, cand.ev, v_allow, level, cand.d_end,
                          analytic, note)

    def _offset_ladder(self, d_start: float, target: TargetOffset) -> List[float]:
        """``d_start`` then steps AWAY from the obstacle until the allowed range ends.

        At most ``offset_ladder_max`` rungs: the search runs inside the 10 Hz planning
        tick, and a wide allowed range (centre line may be crossed) at a fixed 5 cm step
        once cost 1.8 s. The step widens instead, so the whole range is still covered.
        """
        pl = self.cfg.planning
        room = (target.hi - d_start) if target.side > 0 else (d_start - target.lo)
        if room <= 1e-9:
            return [d_start]
        rungs = max(1, int(pl.offset_ladder_max) - 1)
        step = max(pl.offset_step, room / rungs)
        values = [d_start]
        d = d_start + target.side * step
        while target.lo - 1e-9 <= d <= target.hi + 1e-9 and len(values) <= rungs:
            values.append(d)
            d += target.side * step
        return values

    # ----------------------------------------------------------------- entry
    def plan_entry(self, path: OffsetPath, state: VehicleState, pair: ConePair,
                   target: TargetOffset, d_goal: float, cones_sd) -> PlanResult:
        """Lane centre (or any current path state) -> offset beside ``pair``."""
        cfg = self.cfg
        pl = cfg.planning
        first = pair.s_min - cfg.vehicle.front_overhang - cfg.cone_radius  # bumper reaches cone
        last = pair.s_min                                                  # front axle abeam
        ends = list(np.arange(first, last + 1e-6, max(pl.entry_end_step, 0.05)))
        ends = sorted({round(max(e, state.s + 1.0), 3) for e in ends})

        ladder = self._offset_ladder(d_goal, target)
        specs = []
        for d_end in ladder:
            for s1 in ends:
                s0 = max(state.s, s1 - pl.plan_length_max)
                specs.append((path.truncated_before(s0 + 1e-9), s0, s1, d_end))
        return self._pick_by_rung(self._candidates(path, state, specs, cones_sd, (target.lo, target.hi)),
                                  ladder, path, "entry")

    # --------------------------------------------------------------- transit
    def plan_transit(self, path: OffsetPath, state: VehicleState, a: ConePair, b: ConePair,
                     target_b: TargetOffset, d_goal: float, cones_sd,
                     s_min_start: float) -> PlanResult:
        """Direct offset-to-offset transition between two pairs (single quintic)."""
        cfg = self.cfg
        pl = cfg.planning
        gap = b.s_min - a.s_max
        l_body = self.body_free_length(a, b)
        center = 0.5 * ((a.s_max + cfg.front_axle_to_rear + cfg.cone_radius)
                        + (b.s_min - cfg.vehicle.front_overhang - cfg.cone_radius))
        l_lo = max(1.5, l_body)
        l_hi = min(pl.plan_length_max, max(gap + 1.0, l_lo))
        lengths = np.arange(l_lo, l_hi + 1e-6, pl.plan_length_step)
        s_min_start = max(s_min_start, state.s)
        base = path.truncated_before(s_min_start + 1e-9)
        d_from = base.eval(max(s_min_start, base.end_s))[0]
        analytic = self.analytic_speed(l_body, d_goal - d_from)

        spans = set()
        for length in lengths:
            for off in pl.transit_center_offsets:
                s0 = max(center + off - 0.5 * length, s_min_start)
                s1 = min(center + off + 0.5 * length, b.s_min + 0.5)
                spans.add((round(s0, 3), round(s1, 3)))

        ladder = self._offset_ladder(d_goal, target_b)
        specs = [(base, s0, s1, d_end) for d_end in ladder for s0, s1 in sorted(spans)]
        return self._pick_by_rung(self._candidates(base, state, specs, cones_sd, (target_b.lo, target_b.hi)),
                                  ladder, path, "transit", analytic)

    # ---------------------------------------------------------------- return
    def plan_return(self, path: OffsetPath, state: VehicleState, cones_sd,
                    d_goal: float = 0.0, limits=None) -> PlanResult:
        pl = self.cfg.planning
        base = path.truncated_before(state.s + 1e-9)
        lengths = np.arange(pl.return_length, max(pl.plan_length_max, pl.return_length) + 1e-6,
                            pl.plan_length_step)
        cands = self._candidates(base, state, [(base, state.s, state.s + float(L), d_goal) for L in lengths],
                                 cones_sd, limits)
        v_target = self._v_target()
        for cand in cands:  # shortest length that needs no slow-down
            if (cand.lane_ok and cand.steer_ok and cand.v_feas >= v_target
                    and cand.ev.clearance_circle >= pl.min_clearance):
                return self._result(cand, 0, path, "return")
        best = self._pick(cands)
        if best:
            return self._result(best, 0, path, "return")
        cand, level = self._fallback(cands)
        return self._result(cand, level, path, "return")

    # ------------------------------------------------ in-progress re-solve
    def resolve(self, path: OffsetPath, state: VehicleState, s_end: float, d_end: float,
                cones_sd, v_cmd: float, limits=None) -> Optional[PlanResult]:
        """Re-solve the REMAINING part of a maneuver; its end station stays fixed.

        Returns None when the existing path must be kept (spec 7.6).
        """
        cfg = self.cfg
        cand = self._candidate(path.truncated_before(state.s + 1e-9), state, state.s, s_end,
                               d_end, cones_sd, limits)
        if cand is None or not cand.lane_ok or not cand.steer_ok or cand.v_feas < v_cmd * 0.98:
            return None  # remaining distance cannot absorb the change
        current = evaluate(path, state.s, state.psi, cones_sd, cfg)
        clr = cand.ev.clearance_circle
        if clr < cfg.planning.min_clearance and clr <= current.clearance_circle + 0.01:
            return None
        return self._result(cand, 0, path, "resolve")
"""Mission FSM + planning orchestration (pure Python, shared by the node and the simulator).

States: IDLE -> READY -> CRUISE -> AVOID_1 -> TRANSIT -> AVOID_2 -> RETURN -> CRUISE -> DONE

  * Avoidance STARTS when YOLO has seen a cone for ``mission.yolo_trigger_frames``
    consecutive frames (LiDAR must already hold a confirmed track to plan with).
    If YOLO is silent, LiDAR range is the fallback so the start is never missed.
  * AVOID_n ends on geometry AND YOLO (spec 8); YOLO alone never ends it.
  * DONE: all expected pairs handled, lane centre regained, and no obstacle seen
    for ``termination.done_miss_frames`` frames (YOLO / LiDAR per done_sensor_mode).
  * Speed is never commanded to zero after the start command.
"""


import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .perception_node import ConePair, group_pairs

IDLE, READY, CRUISE, TRANSIT, RETURN, DONE = "IDLE", "READY", "CRUISE", "TRANSIT", "RETURN", "DONE"


@dataclass
class PairPlan:
    index: int
    pair: ConePair
    target: TargetOffset
    d_filtered: float
    lane_c: float = 0.0
    planned_sd: Dict[int, tuple] = field(default_factory=dict)

    def snapshot(self) -> None:
        self.planned_sd = {tid: (float(s), float(d))
                           for tid, s, d in zip(self.pair.track_ids, self.pair.s, self.pair.d)}

    def moved(self) -> float:
        """Largest cone displacement since the last plan; inf if the cone set changed."""
        if set(self.planned_sd) != set(self.pair.track_ids):
            return math.inf
        return max(math.hypot(s - self.planned_sd[tid][0], d - self.planned_sd[tid][1])
                   for tid, s, d in zip(self.pair.track_ids, self.pair.s, self.pair.d))


class Mission:
    def __init__(self, cfg):
        self.cfg = cfg
        self.planner = Planner(cfg)
        self.reset()

    def reset(self) -> None:
        self.state = IDLE
        self.ref: Optional[ReferenceLine] = None
        self.path = OffsetPath()
        self.cur: Optional[PairPlan] = None
        self.nxt: Optional[PairPlan] = None
        self.entry_end: Optional[float] = None
        self.transit_span: Optional[tuple] = None
        self.return_end: Optional[float] = None
        self.pairs_done = 0
        self.handled_s = -math.inf
        self.v_limits: Dict[str, tuple] = {}     # name -> (v_allow, s_until)
        self.zero_done = False
        self.start_requested = False
        self.rtk_ok = True
        self.external_fault = False
        self.yolo_hits = 0
        self.yolo_misses = 0
        self.yolo_last_time = -math.inf
        self.lidar_misses = 0
        self.lidar_last_time = -math.inf
        self.last_replan_time = -math.inf
        self.last_plan: Optional[PlanResult] = None
        self.events: List[str] = []
        self.flags: Dict[str, bool] = {}
        self.trigger_source = ""
        self._state = VehicleState(0.0, 0.0, 0.0, 0.0)
        self._raw_ahead = 0
        self._motion_start = None
        self._idle_since: Optional[float] = None
        self._reference_checked = False
        self.lane_c: Optional[float] = None

    # ------------------------------------------------------------- inputs
    def on_zero_done(self) -> None:
        self.zero_done = True

    def on_start(self) -> None:
        self.start_requested = True

    def on_yolo(self, cone_count: int, now: float) -> None:
        self.yolo_last_time = now
        if cone_count > 0:
            self.yolo_hits += 1
            self.yolo_misses = 0
        else:
            self.yolo_misses += 1
            self.yolo_hits = 0

    def on_lidar_frame(self, raw_xy: Sequence[Sequence[float]], now: float) -> None:
        """Raw (unconfirmed) detections of one LiDAR frame, global frame."""
        self.lidar_last_time = now
        ahead = 0
        if self.ref is not None and len(raw_xy):
            sd = self.ref.to_frenet_array(np.asarray(raw_xy, dtype=float))
            ahead = int(np.sum((sd[:, 0] > self._state.s) & (np.abs(sd[:, 1]) <= self._road_gate())))
        self._raw_ahead = ahead
        self.lidar_misses = 0 if ahead else self.lidar_misses + 1

    def _road_gate(self) -> float:
        return self.cfg.road.width / 2.0 + self.cfg.mission.lane_gate_margin

    def yolo_available(self, now: float) -> bool:
        return now - self.yolo_last_time <= self.cfg.mission.yolo_timeout

    def lidar_available(self, now: float) -> bool:
        return now - self.lidar_last_time <= 1.0

    # --------------------------------------------------------------- tick
    def update(self, now: float, x_f: float, y_f: float, yaw: float, v: float,
               tracks: Sequence[Sequence[float]], base_xy=None) -> dict:
        """One 10 Hz planning tick.

        ``x_f, y_f``: FRONT axle in the global frame. ``tracks``: confirmed cones
        as rows (track_id, x, y) in the global frame. ``base_xy``: odometry position
        itself (no lever arm, so independent of yaw) — used by reference mode "motion".
        """
        cfg = self.cfg
        self.events = []
        if self.state == IDLE:
            # Default: nothing gates the start — the launch itself is the trigger.
            # require_zero_done waits for the steering zero (0x202), at most
            # zero_wait_timeout seconds; both are off unless enabled in the YAML.
            if self._idle_since is None:
                self._idle_since = now
            if self.zero_done or not cfg.mission.require_zero_done:
                self._go(READY)
            elif now - self._idle_since >= cfg.mission.zero_wait_timeout:
                self.flags["zero_done_missing"] = True
                self.events.append(f"no 0x202 after {now - self._idle_since:.1f} s, starting anyway")
                self._go(READY)
        if self.state == READY and (self.start_requested or cfg.mission.auto_start):
            self._go(CRUISE)
        if self.state != IDLE and self.state != READY and self.ref is None:
            self._fix_reference(x_f, y_f, yaw, base_xy if base_xy is not None else (x_f, y_f))
        if self.ref is None:
            return self.status(now)

        if base_xy is not None:
            self._recheck_reference(base_xy)
        s, d = self.ref.to_frenet(x_f, y_f)
        st = VehicleState(s, d, self.ref.relative_yaw(yaw), v)
        self._state = st
        groups, cones_sd = self._group(tracks)
        self._refresh(groups)

        if self.state in (CRUISE, RETURN):
            self._maybe_start_avoid(now, st, groups, cones_sd)
        if self.state.startswith("AVOID"):
            self._tick_avoid(now, st, groups, cones_sd)
        elif self.state == TRANSIT:
            self._tick_transit(now, st, cones_sd)
        if self.state == RETURN and self.return_end is not None and st.s >= self.return_end:
            self._go(CRUISE)
        if self.state == CRUISE and self._mission_complete(now):
            self._go(DONE)

        self.v_limits = {k: v_ for k, v_ in self.v_limits.items() if st.s <= v_[1]}
        return self.status(now)

    # ------------------------------------------------------------ helpers
    def _go(self, state: str) -> None:
        self.events.append(f"{self.state}->{state}")
        self.state = state

    def _fix_reference(self, x_f, y_f, yaw, base_xy) -> None:
        """Fix the straight reference line (once).

        fixed       surveyed origin / heading from the YAML
        start_pose  the vehicle yaw at the start — no waiting. Measured on
                    0919_avoidance_f: the EKF yaw is within 0.5 deg even at rest
                    (dual antenna), better than a heading derived from 3 m of travel
                    (0.6 deg). ``recheck_distance`` re-fixes it if the direction the
                    vehicle actually took disagrees; the lane offset itself is anchored
                    on the cone rows later, so only the direction matters here.
        motion      wait for ``motion_distance`` of travel and use that direction, for a
                    vehicle whose yaw estimate cannot be trusted before it moves.
        """
        r = self.cfg.reference
        off = self.cfg.mission.start_lateral_offset
        if r.mode == "fixed":
            self.ref = ReferenceLine(r.origin_x, r.origin_y, math.radians(r.heading_deg))
        elif r.mode == "start_pose":
            # Origin is the lane centre abeam the start pose.
            self.ref = ReferenceLine(x_f + off * math.sin(yaw), y_f - off * math.cos(yaw), yaw)
            self._motion_start = (float(base_xy[0]), float(base_xy[1]))
        else:
            if self._motion_start is None:
                self._motion_start = (float(base_xy[0]), float(base_xy[1]))
                self.events.append("reference: waiting for motion")
                return
            heading = self._travelled_heading(base_xy)
            if heading is None:
                return
            dist = math.hypot(base_xy[0] - self._motion_start[0], base_xy[1] - self._motion_start[1])
            agree = abs(wrap(yaw - heading))
            if agree > math.radians(r.yaw_agree_deg) and dist < r.motion_distance_max:
                self.flags["yaw_not_converged"] = True
                return                      # keep rolling straight until the EKF settles
            self.flags["yaw_not_converged"] = agree > math.radians(r.yaw_agree_deg)
            x0, y0 = self._motion_start
            self.ref = ReferenceLine(x0 + off * math.sin(heading), y0 - off * math.cos(heading), heading)
            self.events.append(f"reference fixed from motion: heading={math.degrees(heading):.1f} deg "
                               f"after {dist:.1f} m (yaw differs {math.degrees(agree):.1f} deg)")
        self.path = OffsetPath()

    def _travelled_heading(self, base_xy) -> Optional[float]:
        """Direction actually travelled since the start, once far enough to be meaningful."""
        if self._motion_start is None:
            return None
        dx = base_xy[0] - self._motion_start[0]
        dy = base_xy[1] - self._motion_start[1]
        r = self.cfg.reference
        need = r.recheck_distance if r.mode == "start_pose" else r.motion_distance
        if math.hypot(dx, dy) < need:
            return None
        return math.atan2(dy, dx)

    def _recheck_reference(self, base_xy) -> None:
        """start_pose only: catch a yaw that was wrong at standstill.

        Runs until the first group is locked — after that the reference must stay put,
        because the planned path and the maneuver end point live in its coordinates.
        """
        r = self.cfg.reference
        if r.mode != "start_pose" or self._reference_checked or self.cur is not None:
            return
        heading = self._travelled_heading(base_xy)
        if heading is None:
            return
        self._reference_checked = True
        error = wrap(heading - self.ref.heading)
        if abs(error) <= math.radians(r.yaw_agree_deg):
            return
        self.flags["reference_refixed"] = True
        off = self.cfg.mission.start_lateral_offset
        x0, y0 = self._motion_start
        self.ref = ReferenceLine(x0 + off * math.sin(heading), y0 - off * math.cos(heading), heading)
        self.path = OffsetPath()
        self.lane_c = None
        self.events.append(f"reference re-fixed: travelled direction differs from start yaw by "
                           f"{math.degrees(error):+.1f} deg -> heading={math.degrees(heading):.1f} deg")

    def _group(self, tracks):
        rows = np.asarray(tracks, dtype=float).reshape(-1, 3)
        if not len(rows):
            return [], np.zeros((0, 2))
        sd = self.ref.to_frenet_array(rows[:, 1:3])
        m = self.cfg.mission
        # Until a group has anchored the lane, the reference line may have drifted by a
        # metre (fixed 30+ m back): gate loosely. Afterwards gate around the known centre.
        if self.lane_c is None:
            keep = np.abs(sd[:, 1]) <= self.cfg.road.width / 2.0 + m.lane_gate_loose
        else:
            keep = np.abs(sd[:, 1] - self.lane_c) <= self.cfg.road.width / 2.0 + m.lane_gate_margin
        ids, sd = rows[keep, 0].astype(int), sd[keep]
        windows = group_pairs(ids, sd, self.cfg.perception.pair_group_window)
        groups = [g for g in (self._row(w) for w in windows) if g is not None]
        return groups, sd

    def _row(self, window: ConePair) -> Optional[ConePair]:
        """The obstacle row inside one longitudinal window, or None for clutter.

        An obstacle group is a short lateral row of cones standing shoulder to shoulder.
        Vegetation that survives the detector shows up as scattered or sprawling
        clusters, so: split by lateral gaps, drop runs longer than a group can be, and
        keep the run with the most cones.
        """
        m = self.cfg.mission
        order = np.argsort(window.d)
        d = window.d[order]
        runs, start = [], 0
        for i in range(1, len(d) + 1):
            if i == len(d) or d[i] - d[i - 1] > m.row_gap_max:
                runs.append(order[start:i])
                start = i
        max_len = m.cones_per_group * self.cfg.perception.cone_diameter + m.row_len_slack
        runs = [r for r in runs if len(r) >= m.row_min_cones
                and window.d[r].max() - window.d[r].min() <= max_len]
        if not runs:
            return None
        center = self.lane_c if self.lane_c is not None else 0.0
        best = max(runs, key=lambda r: (len(r), -float(np.abs(window.d[r] - center).min())))
        return ConePair([window.track_ids[i] for i in best], window.s[best].copy(), window.d[best].copy())

    def _stands(self, index: int) -> Optional[str]:
        sides = list(self.cfg.mission.group_sides)
        side = sides[index - 1] if 0 < index <= len(sides) else None
        return side if side in ("left", "right") else None

    def _lane_for(self, pair: ConePair, index: int) -> Optional[float]:
        """Lane centre anchored on this group; None when it cannot be group ``index``.

        A group on the wrong side for its number is not that group: most likely the
        real one was not detected yet. Taking it anyway shifts the numbering and the
        vehicle passes on the wrong side of everything that follows.
        """
        stands = self._stands(index)
        if stands is None:
            return self.lane_c if self.lane_c is not None else 0.0
        ref_c = self.lane_c if self.lane_c is not None else 0.0
        on_left = float(np.mean(pair.d)) > ref_c
        if on_left != (stands == "left"):
            return None
        c = self.planner.lane_center_from(pair, stands)
        m = self.cfg.mission
        if self.lane_c is None:
            return c if abs(c) <= m.lane_shift_max else None
        return c if abs(c - self.lane_c) <= m.lane_consistency else None

    def _target(self, pair: ConePair, index: int, lane_c: float) -> TargetOffset:
        return self.planner.target_offset(pair, index, lane_c, self._stands(index))

    def _refresh(self, groups: List[ConePair]) -> None:
        """Re-attach the latest cone estimates to the pairs being handled."""
        for plan in (self.cur, self.nxt):
            if plan is None:
                continue
            match = [g for g in groups if set(g.track_ids) & set(plan.pair.track_ids)]
            if not match:
                near = [g for g in groups
                        if abs(g.s_center - plan.pair.s_center) <= self.cfg.perception.pair_group_window]
                match = near[:1]
            if match:
                plan.pair = match[0]

    def _upcoming(self, groups: List[ConePair], after_s: float) -> Optional[ConePair]:
        """Next group ahead that actually BLOCKS the lane-centre path.

        Roadside clutter just inside the gate needs no maneuver; counting it as an
        obstacle group would also shift the group numbering the lane rules rely on.
        """
        index = (self.cur.index if self.cur is not None else self.pairs_done) + 1
        for g in groups:
            if g.s_min <= after_s:
                continue
            lane_c = self._lane_for(g, index)
            if lane_c is not None and self.planner.blocks(g, lane_c):
                return g
        return None

    def _v_now(self) -> float:
        limits = [v for v, _ in self.v_limits.values()]
        return min([min(self.cfg.speed.target, self.cfg.speed.max_clamp)] + limits)

    def _committed(self, st: VehicleState, plan: PairPlan) -> bool:
        bumper = st.s + self.cfg.vehicle.front_overhang
        return plan.pair.s_min - bumper <= self.cfg.replan.commit_distance

    def _adopt(self, result: PlanResult, name: str, s_until: float) -> bool:
        """Take a freshly searched plan. Even a level-3 plan is adopted: for a new
        maneuver the alternative is driving straight at the cones."""
        self.last_plan = result
        if result.segment is None:
            self.events.append(f"plan:{name}:NO_CANDIDATE {result.note}")
            self.flags["plan_failed"] = True
            self.v_limits[name] = (self.cfg.failure.fail_min_speed, s_until)
            return False
        if result.level >= 2:
            self.flags["clearance_warning"] = True
        if result.level >= 3:
            self.flags["steer_infeasible"] = True
        self.path = result.path
        if self.cfg.speed.auto_from_geometry:
            self.v_limits[name] = (result.v_allow, result.segment.s1)
        ev = result.evaluation
        self.events.append(
            f"plan:{name} L={result.segment.length:.2f} s=[{result.segment.s0:.2f},{result.segment.s1:.2f}] "
            f"d_end={result.d_end:+.3f} v_allow={result.v_allow:.2f} clr_c={ev.clearance_circle:.3f} "
            f"clr_r={ev.clearance_rect:.3f} level={result.level}")
        return True

    # -------------------------------------------------------- CRUISE/RETURN
    def _maybe_start_avoid(self, now, st, groups, cones_sd) -> None:
        cfg = self.cfg
        pair = self._upcoming(groups, max(self.handled_s, st.s))
        if pair is None:
            return
        index = self.pairs_done + 1
        lane_c = self._lane_for(pair, index)
        target = self._target(pair, index, lane_c)
        complete = len(pair.track_ids) >= cfg.mission.pair_min_cones
        yolo = (complete and self.yolo_available(now)
                and self.yolo_hits >= cfg.mission.yolo_trigger_frames)
        # Start as soon as a real maneuver fits in front of the group, not at the last
        # possible moment: an earlier start means a longer, gentler entry, more time to
        # slow down, and a path on screen while there is still room to check it.
        # Independent of the commanded cruise speed — lowering speed.target must not
        # postpone the plan (the closed-form length shrinks with speed).
        trigger_speed = max(st.v, cfg.speed.max_clamp)
        need = max(
            self.planner.analytic_length(target.d - st.d, trigger_speed)
            + cfg.vehicle.front_overhang + cfg.cone_radius + cfg.mission.lidar_fallback_margin,
            cfg.mission.trigger_distance_min,
        )
        lidar = pair.s_min - st.s <= need
        if not (yolo or lidar):
            return
        self.trigger_source = "yolo" if yolo else "lidar_fallback"
        self.flags["yolo_trigger_missed"] = not yolo

        self.cur = PairPlan(index, pair, target, target.d, lane_c)
        if self.lane_c is None:
            self.events.append(f"lane anchored on group {index}: centre d={lane_c:+.2f}")
        self.lane_c = lane_c
        self.nxt = None
        self.transit_span = None
        self.return_end = None
        self.v_limits.pop("return", None)
        result = self.planner.plan_entry(self.path, st, pair, target, target.d,
                                         self._cones_until(cones_sd, pair.s_max))
        self._adopt(result, "entry", pair.s_max)
        if result.segment is not None and result.level < 3:
            self.entry_end = result.segment.s1
            self.cur.d_filtered = result.d_end
        else:
            self.entry_end = st.s
        self.cur.snapshot()
        self.last_replan_time = now
        self._go(f"AVOID_{self.cur.index}")
        self._try_plan_transit(now, st, groups, cones_sd)

    # ----------------------------------------------------------------- AVOID
    def _try_plan_transit(self, now, st, groups, cones_sd, force: bool = False) -> None:
        cfg = self.cfg
        if self.nxt is None:
            pair = self._upcoming(groups, self.cur.pair.s_max)
            if pair is None or pair.s_min - self.cur.pair.s_max >= cfg.planning.transit_threshold_gap:
                return
            lane_c = self._lane_for(pair, self.cur.index + 1)
            target = self._target(pair, self.cur.index + 1, lane_c)
            self.nxt = PairPlan(self.cur.index + 1, pair, target, target.d, lane_c)
        elif not force:
            return
        result = self.planner.plan_transit(
            self.path, st, self.cur.pair, self.nxt.pair, self.nxt.target, self.nxt.d_filtered,
            self._cones_until(cones_sd, self.nxt.pair.s_max), max(self.entry_end or st.s, st.s))
        if self._adopt(result, "transit", self.nxt.pair.s_max):
            self.transit_span = (result.segment.s0, result.segment.s1)
            self.nxt.d_filtered = result.d_end
        else:
            self.transit_span = None
        self.nxt.snapshot()
        self.cur.snapshot()
        self.last_replan_time = now

    def _cones_until(self, cones_sd, s_end: float) -> np.ndarray:
        """Every tracked cone the maneuver drives past — whoever "owns" it.

        Groups BEYOND the maneuver are left out: a maneuver ends by holding its offset,
        which scored against a later group looks like a collision and pushed the planner
        into needless extra offset. Cones before that point always count, including a
        group that needed no maneuver of its own.
        """
        cones_sd = np.asarray(cones_sd, dtype=float).reshape(-1, 2)
        return cones_sd[cones_sd[:, 0] <= s_end + 0.5 * self.cfg.perception.pair_group_window]

    def _replan_due(self, now, plan: PairPlan) -> bool:
        """Cone estimate moved, or the filtered offset has not yet reached the raw one.

        The second clause matters: the filter only advances on a re-plan, so without
        it a changed target would be approached once and then left stranded.
        """
        r = self.cfg.replan
        if not r.enabled or now - self.last_replan_time < r.period:
            return False
        lag = abs(self._target(plan.pair, plan.index, self._plan_lane(plan)).d - plan.d_filtered)
        return plan.moved() >= r.pos_threshold or lag > r.offset_tolerance

    def _plan_lane(self, plan: PairPlan) -> float:
        lane_c = self._lane_for(plan.pair, plan.index)
        return plan.lane_c if lane_c is None else lane_c

    def _filtered_target(self, plan: PairPlan) -> float:
        a = self.cfg.replan.offset_filter_alpha
        plan.lane_c = self._plan_lane(plan)
        new = self._target(plan.pair, plan.index, plan.lane_c)
        # A cone joining/leaving the pair (or a side flip) is a step change in what we
        # know, not noise: take it at once instead of low-passing toward it.
        step_change = (set(plan.planned_sd) != set(plan.pair.track_ids)
                       or new.side != plan.target.side)
        plan.target = new
        if step_change:
            return new.d
        # Offsets grown by the clearance ladder are kept: never filter back below them.
        blended = (1.0 - a) * plan.d_filtered + a * new.d
        return blended

    def _tick_avoid(self, now, st, groups, cones_sd) -> None:
        cfg = self.cfg
        cur = self.cur
        in_entry = self.entry_end is not None and st.s < self.entry_end

        if in_entry and self._replan_due(now, cur) and not self._committed(st, cur):
            d_new = self._filtered_target(cur)
            result = self.planner.resolve(self.path, st, self.entry_end, d_new,
                                           self._cones_until(cones_sd, cur.pair.s_max), self._v_now(),
                                           (cur.target.lo, cur.target.hi))
            self.last_replan_time = now
            cur.snapshot()
            if result is not None:
                # Later segments were built on the old offset; rebuild the transit on the new one.
                result.path = result.path.truncated_before(self.entry_end)
                self._adopt(result, "entry", cur.pair.s_max)
                cur.d_filtered = d_new
                if self.nxt is not None:
                    self._try_plan_transit(now, st, groups, cones_sd, force=True)
            else:
                self.events.append("replan:entry rejected, keeping path")

        transit_pending = self.transit_span is None or st.s < self.transit_span[0]
        if self.nxt is None:
            self._try_plan_transit(now, st, groups, cones_sd)
        elif (transit_pending and not self._committed(st, self.nxt)
              and (self._replan_due(now, self.nxt) or self.transit_span is None)):
            self.nxt.d_filtered = self._filtered_target(self.nxt)
            self._try_plan_transit(now, st, groups, cones_sd, force=True)

        if self.nxt is not None and self.transit_span is not None:
            if not in_entry:
                self._go(TRANSIT)
            return

        passed = st.s >= cur.pair.s_max + cfg.front_axle_to_rear + cfg.planning.clear_margin
        # YOLO keeps seeing the NEXT pair when one is tracked ahead, so its silence
        # is only required when nothing else is ahead. Geometry is always required.
        more_ahead = self._upcoming(groups, cur.pair.s_max) is not None
        yolo_clear = (not self.yolo_available(now) or more_ahead
                      or self.yolo_misses >= cfg.termination.pass_miss_frames)
        if passed and yolo_clear:
            self.pairs_done += 1
            self.handled_s = cur.pair.s_max
            result = self.planner.plan_return(self.path, st, self._cones_until(cones_sd, cur.pair.s_max),
                                              d_goal=cur.lane_c,
                                              limits=self.planner.limits(cur.index, cur.lane_c))
            self._adopt(result, "return", st.s + cfg.planning.plan_length_max)
            self.return_end = result.segment.s1 if result.segment is not None else st.s
            self.cur = None
            self.entry_end = None
            self._go(RETURN)

    # --------------------------------------------------------------- TRANSIT
    def _tick_transit(self, now, st, cones_sd) -> None:
        nxt = self.nxt
        s0, s1 = self.transit_span
        if st.s >= s1:
            self.pairs_done += 1
            self.handled_s = self.cur.pair.s_max
            self.cur, self.nxt = nxt, None
            self.lane_c = self.cur.lane_c
            self.entry_end = None
            self.transit_span = None
            self._go(f"AVOID_{self.cur.index}")
            return
        if not self._replan_due(now, nxt) or self._committed(st, nxt):
            return
        self.last_replan_time = now
        d_new = self._filtered_target(nxt)
        nxt.snapshot()
        if st.s < s0:
            self._try_plan_transit(now, st, [], cones_sd, force=True)
            return
        result = self.planner.resolve(self.path, st, s1, d_new,
                                       self._cones_until(cones_sd, nxt.pair.s_max), self._v_now(),
                                       (nxt.target.lo, nxt.target.hi))
        if result is not None:
            self._adopt(result, "transit", nxt.pair.s_max)
            nxt.d_filtered = d_new
        else:
            self.events.append("replan:transit rejected, keeping path")

    # ------------------------------------------------------------------ DONE
    def _mission_complete(self, now) -> bool:
        cfg = self.cfg
        if self.pairs_done < cfg.mission.expected_pairs:
            return False
        n = cfg.termination.done_miss_frames
        checks = []
        mode = cfg.termination.done_sensor_mode
        if mode in ("both", "either", "yolo") and self.yolo_available(now):
            checks.append(self.yolo_misses >= n)
        if mode in ("both", "either", "lidar") and self.lidar_available(now):
            checks.append(self.lidar_misses >= n)
        if not checks:
            return True  # no sensor alive to object; geometry already satisfied
        return any(checks) if mode == "either" else all(checks)

    # ---------------------------------------------------------------- output
    def speed_command(self) -> float:
        cfg = self.cfg
        if self.state in (IDLE, READY):
            return 0.0
        v = self._v_now()
        if not self.rtk_ok:
            v = min(v, cfg.speed.rtk_degraded)
        if self.external_fault:
            v = min(v, cfg.speed.fault)
        floor = min(cfg.failure.fail_min_speed, cfg.speed.rtk_degraded, cfg.speed.fault)
        return max(v, floor) if cfg.failure.never_stop else v

    def status(self, now: float) -> dict:
        st = self._state
        ev = self.last_plan.evaluation if self.last_plan and self.last_plan.evaluation else None
        seg = self.last_plan.segment if self.last_plan else None
        d_path = self.path.eval(st.s)[0] if self.ref is not None else 0.0
        return {
            "t": now,
            "state": self.state,
            "s": st.s, "d": st.d, "psi": st.psi, "v": st.v,
            "d_path": d_path,
            "pairs_done": self.pairs_done,
            "lane_c": self.lane_c if self.lane_c is not None else math.nan,
            "trigger": self.trigger_source,
            "cur_d_raw": self.cur.target.d_raw if self.cur else math.nan,
            "cur_d_target": self.cur.d_filtered if self.cur else math.nan,
            "nxt_d_raw": self.nxt.target.d_raw if self.nxt else math.nan,
            "nxt_d_target": self.nxt.d_filtered if self.nxt else math.nan,
            "clipped": bool((self.cur and self.cur.target.clipped) or (self.nxt and self.nxt.target.clipped)),
            "plan_L": seg.length if seg else math.nan,
            "plan_level": self.last_plan.level if self.last_plan else 0,
            "v_allow": self.last_plan.v_allow if self.last_plan else math.nan,
            "v_allow_analytic": self.last_plan.v_allow_analytic if self.last_plan else math.nan,
            "clearance_circle": ev.clearance_circle if ev else math.nan,
            "clearance_rect": ev.clearance_rect if ev else math.nan,
            "v_cmd": self.speed_command(),
            "rtk_ok": self.rtk_ok,
            "yolo_hits": self.yolo_hits, "yolo_misses": self.yolo_misses,
            "lidar_misses": self.lidar_misses,
            "flags": [k for k, v in self.flags.items() if v],
            "events": list(self.events),
        }
"""FSM + path planning node (10 Hz)."""


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
        self.faults = {"control": False}
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
