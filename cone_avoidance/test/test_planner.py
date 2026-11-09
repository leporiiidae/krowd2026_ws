import math

import numpy as np
import pytest

from cone_avoidance.mission_node import OffsetPath
from cone_avoidance.params import Config
from cone_avoidance.mission_node import Planner, VehicleState
from cone_avoidance.perception_node import ConePair


def pair(s, ds, first_id=0):
    return ConePair(list(range(first_id, first_id + len(ds))), np.full(len(ds), float(s)), np.array(ds, float))


# Three 0.45 m cones touching each other, flush against a lane boundary (road half 1.9).
LEFT = [1.675, 1.225, 0.775]       # 1st group: against the centre line
RIGHT = [-0.775, -1.225, -1.675]   # 2nd group: against the solid line


def test_derived_widths():
    cfg = Config()
    assert cfg.collision_width == pytest.approx(1.18)
    assert cfg.d_limit == pytest.approx(1.31)
    wide_tyres = Config({"vehicle": {"body_width": 1.0, "track_width": 1.0, "tire_width": 0.25}})
    assert wide_tyres.collision_width == pytest.approx(1.25)


def test_target_offsets_for_the_three_cone_groups():
    planner = Planner(Config())
    reach = 0.225 + 0.59 + 0.30
    t = planner.target_offset(pair(10, LEFT), 1)
    assert (t.side, t.clipped) == (-1, False) and t.d == pytest.approx(0.775 - reach)
    t = planner.target_offset(pair(15, RIGHT, 3), 2)
    assert (t.side, t.clipped) == (+1, False) and t.d == pytest.approx(-0.775 + reach)


def test_centre_line_may_be_crossed_only_for_listed_groups():
    cfg = Config()
    planner = Planner(cfg)
    lo1, hi1 = planner.limits(1)
    lo2, hi2 = planner.limits(2)
    assert (lo1, hi1) == pytest.approx((-1.31, 1.31))
    assert lo2 == pytest.approx(-1.31)                             # solid line: never
    assert hi2 == pytest.approx(1.31 + cfg.road.center_cross_extra)  # centre line: allowed
    wide = pair(15, [0.3, -0.15, -0.6], 3)                         # a group reaching past the centre
    need = 0.3 + 0.225 + 0.59 + 0.30
    assert need > 1.31
    t1, t2 = planner.target_offset(wide, 1), planner.target_offset(wide, 2)
    assert t1.clipped and t1.d == pytest.approx(1.31)
    assert not t2.clipped and t2.d == pytest.approx(need)
    # the clearance ladder may grow toward the centre line only as far as allowed
    assert planner._offset_ladder(t2.d, t2)[-1] <= hi2 + 1e-9


def test_lane_hard_constraint_off_does_not_clip():
    far_in = pair(10, [-1.0, 0.4])
    hard = Planner(Config()).target_offset(far_in)
    assert hard.clipped and abs(hard.d) == pytest.approx(1.31)
    soft = Planner(Config({"road": {"lane_hard_constraint": False}})).target_offset(far_in)
    assert not soft.clipped and abs(soft.d) > 1.31


def test_body_free_length_is_about_1p1_m_for_4m_gap():
    planner = Planner(Config())
    assert planner.body_free_length(pair(10, LEFT), pair(14, RIGHT)) == pytest.approx(4 - 2.914, abs=1e-3)


def test_entry_plan_clears_cones_and_stays_in_lane():
    cfg = Config()
    planner = Planner(cfg)
    p = pair(12, LEFT)
    target = planner.target_offset(p)
    res = planner.plan_entry(OffsetPath(), VehicleState(0.0, 0.0, 0.0, 1.0), p, target, target.d, p.sd)
    assert res.level == 0 and res.evaluation.clearance_circle >= 0.0
    assert res.segment.s1 <= p.s_min + 1e-6            # offset reached no later than abeam
    assert abs(res.d_end) <= cfg.d_limit
    assert res.v_allow == pytest.approx(cfg.speed.target)


@pytest.mark.parametrize("gap", [4.0, 5.0, 6.0])
def test_transit_is_single_quintic_between_offsets(gap):
    cfg = Config()
    planner = Planner(cfg)
    a, b = pair(10, LEFT), pair(10 + gap, RIGHT, 3)
    cones = np.vstack((a.sd, b.sd))
    ta, tb = planner.target_offset(a, 1), planner.target_offset(b, 2)
    st = VehicleState(0.0, 0.0, 0.0, 1.0)
    entry = planner.plan_entry(OffsetPath(), st, a, ta, ta.d, cones)
    res = planner.plan_transit(entry.path, st, a, b, tb, tb.d, cones, entry.segment.s1)
    assert res.evaluation.clearance_circle >= 0.0
    assert len(res.path.segments) == 2                  # entry + ONE transit segment
    assert res.path.eval(res.segment.s0)[0] == pytest.approx(entry.d_end, abs=1e-6)
    assert res.path.eval(res.segment.s1)[0] == pytest.approx(res.d_end)
    assert entry.d_end < 0 < res.d_end                  # right of group 1, then left of group 2


def test_slower_stepper_lowers_the_allowed_speed_not_the_safety():
    fast = Config({"speed": {"target": 1.5}, "steering": {"max_rate_deg_s": 60.0}})
    slow = Config({"speed": {"target": 1.5}, "steering": {"max_rate_deg_s": 8.0}})
    out = []
    for cfg in (fast, slow):
        planner = Planner(cfg)
        a, b = pair(10, LEFT), pair(14, RIGHT, 3)
        cones = np.vstack((a.sd, b.sd))
        st = VehicleState(0.0, 0.0, 0.0, 1.0)
        ta, tb = planner.target_offset(a, 1), planner.target_offset(b, 2)
        entry = planner.plan_entry(OffsetPath(), st, a, ta, ta.d, cones)
        out.append(planner.plan_transit(entry.path, st, a, b, tb, tb.d, cones, entry.segment.s1))
    assert out[1].v_allow < out[0].v_allow
    assert out[1].v_allow >= slow.failure.fail_min_speed
    assert out[1].evaluation.clearance_circle >= 0.0
    # spec 7.4 closed form with the body removed from the gap is far more pessimistic
    assert out[0].v_allow_analytic < out[0].v_allow


def test_resolve_keeps_end_station_and_is_c2():
    planner = Planner(Config())
    p = pair(12, LEFT)
    target = planner.target_offset(p)
    st0 = VehicleState(0.0, 0.0, 0.0, 1.0)
    entry = planner.plan_entry(OffsetPath(), st0, p, target, target.d, p.sd)
    s_now = 4.0
    d, d1, _ = entry.path.eval(s_now)
    before = entry.path.eval(s_now)
    res = planner.resolve(entry.path, VehicleState(s_now, d, math.atan(d1) * 0.6, 1.0),
                          entry.segment.s1, target.d - 0.08, p.sd, 1.0)
    assert res is not None
    assert res.segment.s1 == pytest.approx(entry.segment.s1)        # end is fixed in space
    assert res.path.eval(s_now + 1e-9) == pytest.approx(before, abs=1e-6)
    assert res.path.eval(entry.segment.s1)[0] == pytest.approx(target.d - 0.08)


def test_resolve_refuses_when_remaining_distance_is_too_short():
    planner = Planner(Config())
    p = pair(12, LEFT)
    target = planner.target_offset(p)
    entry = planner.plan_entry(OffsetPath(), VehicleState(0.0, 0.0, 0.0, 1.0), p, target, target.d, p.sd)
    s_now = entry.segment.s1 - 0.6
    d, d1, _ = entry.path.eval(s_now)
    res = planner.resolve(entry.path, VehicleState(s_now, d, math.atan(d1), 1.0),
                          entry.segment.s1, target.d - 0.5, p.sd, 1.0)
    assert res is None
