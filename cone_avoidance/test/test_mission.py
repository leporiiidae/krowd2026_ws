import math

import pytest

from cone_avoidance.mission_node import CRUISE, DONE, IDLE, READY, RETURN, TRANSIT, Mission
from cone_avoidance.params import Config, deep_merge
from cone_avoidance.sim import Simulator

# 1st group: 3 touching cones on the LEFT (centre line); 2nd: 3 on the RIGHT (solid line).
PAIR1 = [(0, 10.0, 1.675), (1, 10.0, 1.225), (2, 10.0, 0.775)]
PAIR2 = [(3, 15.0, -0.775), (4, 15.0, -1.225), (5, 15.0, -1.675)]


FIXED = {"reference": {"mode": "fixed", "origin_x": 0.0, "origin_y": 0.0, "heading_deg": 0.0}}


def started(cfg=None):
    m = Mission(cfg or Config(FIXED))
    m.on_zero_done()
    m.on_start()
    m.update(0.0, 0.0, 0.0, 0.0, 0.0, [])
    return m


def drive(m, s_from, s_to, tracks, t0=0.0, yolo=None, step=0.1):
    """Ideal follower: front axle exactly on the planned path."""
    t, s = t0, s_from
    while s < s_to:
        if yolo is not None:
            m.on_yolo(yolo(s), t)
        d, d1, _ = m.path.eval(s)
        m.update(t, s, d, math.atan(d1), 1.0, tracks)
        s += step
        t += 0.1
    return t


def test_start_pose_reference_needs_no_wait_and_is_rechecked():
    """Default: steer from the first tick; the start yaw is verified after a few metres."""
    cfg = Config()
    assert cfg.reference.mode == "start_pose"
    wb = cfg.vehicle.wheelbase
    road, wrong = math.radians(-127.0), math.radians(-100.0)
    m = Mission(cfg)
    m.update(0.0, wb * math.cos(wrong), wb * math.sin(wrong), wrong, 0.0, [], (0.0, 0.0))
    assert m.ref is not None                                # no waiting
    assert math.degrees(m.ref.heading) == pytest.approx(-100.0)
    ux, uy = math.cos(road), math.sin(road)
    for k in range(1, 60):                                  # now drive the real direction
        bx, by = ux * 0.1 * k, uy * 0.1 * k
        m.update(0.1 * k, bx + wb * ux, by + wb * uy, road, 1.0, [], (bx, by))
    assert math.degrees(m.ref.heading) == pytest.approx(-127.0, abs=0.5)   # re-fixed
    assert "reference_refixed" in m.status(6.0)["flags"]


def test_start_pose_reference_is_kept_when_the_start_yaw_was_right():
    cfg = Config()
    wb = cfg.vehicle.wheelbase
    road = math.radians(-127.0)
    m = Mission(cfg)
    ux, uy = math.cos(road), math.sin(road)
    for k in range(0, 60):
        bx, by = ux * 0.1 * k, uy * 0.1 * k
        m.update(0.1 * k, bx + wb * ux, by + wb * uy, road, 1.0, [], (bx, by))
    assert math.degrees(m.ref.heading) == pytest.approx(-127.0, abs=0.1)
    assert "reference_refixed" not in m.status(6.0)["flags"]


def test_nothing_gates_the_start_by_default():
    """The launch is the trigger: no start command, no waiting for the steering zero."""
    m = Mission(Config(FIXED))
    m.update(0.0, 0, 0, 0, 0, [])
    assert m.state == CRUISE and m.speed_command() > 0.0


def test_zero_wait_is_available_when_asked_for():
    cfg = Config(deep_merge(FIXED, {"mission": {"require_zero_done": True, "zero_wait_timeout": 3.0}}))
    m = Mission(cfg)
    m.update(0.0, 0, 0, 0, 0, [])
    assert m.state == IDLE and m.speed_command() == 0.0      # zeroing may be running
    m.update(3.5, 0, 0, 0, 0, [])
    assert m.state == CRUISE and "zero_done_missing" in m.status(3.5)["flags"]


def test_manual_start_when_auto_start_is_off():
    m = Mission(Config(deep_merge(FIXED, {"mission": {"auto_start": False}})))
    m.update(0.0, 0, 0, 0, 0, [])
    assert m.state == READY and m.speed_command() == 0.0
    m.on_start()
    m.update(0.1, 0, 0, 0, 0, [])
    assert m.state == CRUISE and m.speed_command() > 0.0


def test_avoidance_starts_on_yolo_before_the_lidar_fallback_distance():
    """YOLO starts it as soon as the group is seen; LiDAR alone waits for
    mission.trigger_distance_min, so put the group beyond that for this test."""
    m = started()
    far = [(i, 20.0, d) for i, d in enumerate((1.675, 1.225, 0.775))]
    m.update(0.1, 0.0, 0.0, 0.0, 1.0, far)             # LiDAR tracks exist, YOLO silent
    assert m.state == CRUISE
    for i in range(3):
        m.on_yolo(2, 0.2 + 0.1 * i)
    m.update(0.5, 0.2, 0.0, 0.0, 1.0, far)
    assert m.state == "AVOID_1" and m.trigger_source == "yolo"
    assert m.path.eval(20.0)[0] == pytest.approx(-0.34, abs=0.06)  # passes on the RIGHT


def test_yolo_trigger_waits_for_all_three_cones_of_the_group():
    m = started()
    partial = [(i, 20.0, d) for i, d in enumerate((1.675, 1.225))]   # beyond the fallback range
    for i in range(5):
        m.on_yolo(2, 0.1 * i)
    m.update(0.5, 0.0, 0.0, 0.0, 1.0, partial)          # inner cone not confirmed yet
    assert m.state == CRUISE


def test_lidar_fallback_starts_avoidance_when_yolo_is_dead():
    m = started()
    drive(m, 0.0, 6.0, PAIR1)
    assert m.state == "AVOID_1" and m.trigger_source == "lidar_fallback"
    assert "yolo_trigger_missed" in m.status(0.0)["flags"]


def test_lidar_fallback_distance_does_not_shrink_with_target_speed():
    """A lower cruise command must not postpone creation of the avoidance path."""
    low = Config(deep_merge(FIXED, {"speed": {"target": 3.0 / 3.6, "max_clamp": 6.0 / 3.6}}))
    nominal = Config(deep_merge(FIXED, {"speed": {"target": 5.0 / 3.6, "max_clamp": 6.0 / 3.6}}))
    a, b = started(low), started(nominal)
    # A larger first manoeuvre makes the steering-rate look-ahead speed-dependent.
    shifted = [(i, x, y + 0.30) for i, x, y in PAIR1]
    for mission in (a, b):
        mission.update(0.1, 3.35, 0.55, 0.0, 0.0, shifted)
    assert a.state == b.state == "AVOID_1"


def test_avoidance_starts_early_enough_to_see_the_path():
    """The plan must exist well before the cones, not only in the last few metres."""
    cfg = Config(FIXED)
    m = started(cfg)
    far = [(i, 11.0, d) for i, d in enumerate((1.675, 1.225, 0.775))]
    m.update(0.1, 0.0, 0.0, 0.0, 0.0, far)                  # 11 m ahead, standing still
    assert m.state == "AVOID_1"
    assert m.path.eval(11.0)[0] == pytest.approx(-0.34, abs=0.08)
    assert cfg.mission.trigger_distance_min >= 10.0


def test_full_sequence_and_termination_needs_geometry():
    cfg = Config(FIXED)
    m = started(cfg)
    tracks = PAIR1 + PAIR2
    seen = lambda s: 2 if s < 14.5 else 0                # YOLO loses the cones once abeam
    t = drive(m, 0.0, 9.0, tracks, yolo=seen)
    assert m.state in ("AVOID_1", TRANSIT)
    assert m.transit_span is not None and m.transit_span[1] <= 15.5
    t = drive(m, 9.0, 15.6, tracks, t, yolo=seen)
    assert m.state == "AVOID_2" and m.pairs_done == 1
    # YOLO has been blind for > pass_miss_frames, but the rear tyre has not cleared the cone.
    t = drive(m, 15.6, 15.0 + cfg.front_axle_to_rear + cfg.planning.clear_margin - 0.2, tracks, t, yolo=seen)
    assert m.state == "AVOID_2" and m.yolo_misses >= cfg.termination.pass_miss_frames
    t = drive(m, m._state.s, 18.0, tracks, t, yolo=seen)
    assert m.state == RETURN and m.pairs_done == 2
    t = drive(m, 18.0, 24.0, tracks, t, yolo=seen)
    assert m.state == DONE
    assert m.path.eval(30.0)[0] == pytest.approx(0.0)
    assert m.path.eval(15.0)[0] == pytest.approx(0.34, abs=0.06)  # LEFT of the 2nd group


def test_done_requires_obstacle_free_frames():
    m = started()
    tracks = PAIR1 + PAIR2
    t = drive(m, 0.0, 17.0, tracks, yolo=lambda s: 2)
    # Geometry satisfied, YOLO still reports cones -> termination is held back by YOLO.
    assert m.state == "AVOID_2"
    m.yolo_last_time = -math.inf                          # YOLO drops out entirely
    t = drive(m, 17.0, 17.5, tracks, t)
    assert m.state == RETURN                              # geometry alone decides then


def test_wide_gap_returns_to_centre_between_pairs():
    far = [(3, 25.0, -0.775), (4, 25.0, -1.225), (5, 25.0, -1.675)]
    m = started()
    seen = lambda s: 2 if (s < 9.5 or 13.0 < s < 24.5) else 0
    t = drive(m, 0.0, 13.5, PAIR1 + far, yolo=seen)
    assert m.state in (RETURN, CRUISE, "AVOID_2") and m.transit_span is None
    drive(m, 13.5, 35.0, PAIR1 + far, t, yolo=seen)
    assert m.state == DONE and m.pairs_done == 2


def test_speed_is_never_zero_after_start_even_with_faults():
    m = started()
    m.rtk_ok = False
    m.external_fault = True
    m.update(1.0, 1.0, 0.0, 0.0, 1.0, PAIR1)
    assert 0.0 < m.speed_command() <= m.cfg.speed.rtk_degraded


def test_motion_reference_ignores_a_wrong_yaw_at_rest():
    """For a vehicle whose yaw is only valid once moving (reference.mode: motion)."""
    m = Mission(Config({"reference": {"mode": "motion"}}))
    m.on_zero_done(); m.on_start()
    road = math.radians(-127.0)
    ux, uy = math.cos(road), math.sin(road)
    wb = 1.34
    t, ref_fixed_at = 0.0, None
    for k in range(80):
        dist = 0.1 * k
        yaw = math.radians(18.0) if dist < 1.5 else road    # garbage at rest, then converged
        bx, by = 40.0 + ux * dist, 44.0 + uy * dist
        m.update(t, bx + wb * math.cos(yaw), by + wb * math.sin(yaw), yaw, 1.0, [], (bx, by))
        if m.ref is not None and ref_fixed_at is None:
            ref_fixed_at = dist
        t += 0.1
    assert ref_fixed_at is not None and ref_fixed_at >= 3.0
    assert math.degrees(m.ref.heading) == pytest.approx(-127.0, abs=0.5)
    assert m.speed_command() > 0.0                          # kept rolling while waiting


def test_roadside_clutter_is_not_an_obstacle_group():
    m = started()
    bush = [(9, 6.0, -2.0), (10, 6.3, -1.95)]               # at the solid line, outside the lane
    for i in range(5):
        m.on_yolo(2, 0.1 * i)
    m.update(0.5, 0.0, 0.0, 0.0, 1.0, bush)
    assert m.state == CRUISE
    drive(m, 0.0, 9.0, bush + PAIR1 + PAIR2, yolo=lambda s: 2)
    assert m.cur is not None and m.cur.index == 1           # numbering not shifted by the bush
    assert set(m.cur.pair.track_ids) == {0, 1, 2}


def test_lane_is_anchored_on_the_groups_not_on_a_drifted_reference():
    """Seen in the 0919 recording: 35 m after the start the reference had drifted ~1 m, so
    the first group straddled d = 0 and the plan went LEFT, across the centre line."""
    drift = -0.95                                           # true lane centre in reference coords
    g1 = [(i, 33.0, d + drift) for i, d in enumerate((1.675, 1.225, 0.775))]
    g2 = [(3 + i, 38.0, d + drift) for i, d in enumerate((-0.775, -1.225, -1.675))]
    veg = [(20 + i, 36.0 + 0.7 * i, -4.0 - 0.6 * i) for i in range(6)]   # roadside growth
    m = started()
    drive(m, 20.0, 31.0, g1 + g2 + veg, yolo=lambda s: 2)
    assert m.lane_c == pytest.approx(drift, abs=0.05)
    assert m.cur.index == 1 and m.cur.target.side == -1     # group 1 stands LEFT -> pass RIGHT
    assert m.path.eval(33.0)[0] == pytest.approx(drift - 0.34, abs=0.08)
    assert m.nxt is not None and m.nxt.target.side == +1
    assert m.path.eval(38.0)[0] == pytest.approx(drift + 0.34, abs=0.10)
    lo, hi = m.planner.limits(1, m.lane_c)
    assert (lo, hi) == pytest.approx((drift - 1.31, drift + 1.31), abs=0.05)   # limits move with the lane


def test_group_on_the_wrong_side_is_not_taken_for_the_next_group():
    """Group 1 not detected yet, group 2 already is: waiting beats passing on the wrong side."""
    m = started()
    for i in range(5):
        m.on_yolo(2, 0.1 * i)
    m.update(0.5, 0.0, 0.0, 0.0, 1.0, PAIR2)             # only the RIGHT group is visible
    assert m.state == CRUISE and m.cur is None
    t = drive(m, 0.0, 9.0, PAIR1 + PAIR2, yolo=lambda s: 2)
    assert m.cur is not None and m.cur.index == 1 and set(m.cur.pair.track_ids) == {0, 1, 2}


def test_maneuver_checks_every_cone_it_drives_past():
    """A group needing no offset of its own must still be cleared by the next maneuver."""
    m = started()
    near_left = [(0, 10.0, 1.80), (1, 10.0, 1.40), (2, 10.0, 1.20)]   # inner edge 0.975: needs no offset
    drive(m, 0.0, 18.0, near_left + PAIR2, yolo=lambda s: 2)
    d_at_first = m.path.eval(10.0)[0]
    assert 1.20 - 0.225 - 0.59 - d_at_first >= 0.25         # not yet swung left beside group 1
    assert m.path.eval(15.0)[0] == pytest.approx(0.34, abs=0.08)


@pytest.mark.parametrize("gap", [4.0, 6.0])
def test_closed_loop_simulation_has_no_collision(gap):
    cones = [[10.0, d] for d in (1.675, 1.225, 0.775)] + [[10.0 + gap, d] for d in (-0.775, -1.225, -1.675)]
    cfg = Config({"sim": {"cones": [v for c in cones for v in c]}})
    for seed in range(3):
        result = Simulator(cfg, seed).run()
        assert result.done and not result.collided
        assert result.min_clearance > 0.10
        assert result.max_lane_excursion <= 0.0
        assert result.min_speed > 0.0
