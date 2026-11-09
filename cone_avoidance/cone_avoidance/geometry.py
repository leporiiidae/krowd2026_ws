"""Coordinate conventions, Frenet transform and quintic offset paths.

Sign conventions (spec section 3) — every module relies on these:
  * global frame: local ENU, yaw counter-clockwise positive
  * Frenet: ``s`` along the straight reference line, ``d`` lateral, LEFT positive
  * wheel steering angle delta: LEFT turn positive
  * control reference point: FRONT axle centre
"""

from __future__ import annotations

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
