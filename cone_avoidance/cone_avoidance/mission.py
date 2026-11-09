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

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from .geometry import OffsetPath, ReferenceLine, wrap
from .planner import Planner, PlanResult, TargetOffset, VehicleState
from .tracker import ConePair, group_pairs

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
