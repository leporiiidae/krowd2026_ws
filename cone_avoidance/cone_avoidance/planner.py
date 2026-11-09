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

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .geometry import OffsetPath, QuinticSegment, QUINTIC_MAX_D2, QUINTIC_MAX_D3
from .swept import Evaluation, evaluate, evaluate_branches
from .tracker import ConePair


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
