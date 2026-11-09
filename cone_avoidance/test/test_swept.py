import math

import numpy as np
import pytest

from cone_avoidance.geometry import OffsetPath, QuinticSegment
from cone_avoidance.params import Config
from cone_avoidance.swept import body_circles, clearance_at_pose, evaluate, trace_path

CFG = Config()


def test_three_circles_cover_body_as_in_spec():
    centers, radius = body_circles(CFG)
    body_len = 2.464
    assert radius == pytest.approx(math.hypot(body_len / 6, 1.18 / 2))
    assert centers == pytest.approx([-0.28 + body_len / 6, -0.28 + body_len / 2, -0.28 + 5 * body_len / 6])


def test_straight_pass_clearance_is_plain_lateral_distance():
    path = OffsetPath(base_offset=0.35)
    cone = np.array([[10.0, -0.765]])
    ev = evaluate(path, 0.0, 0.0, cone, CFG)
    lateral = 0.35 + 0.765
    assert ev.clearance_rect == pytest.approx(lateral - 0.59 - 0.225, abs=1e-3)   # = lateral_margin 0.30
    # circle approximation is conservative by (r - half width) = ~0.156 m when abeam
    assert ev.clearance_rect - ev.clearance_circle == pytest.approx(body_circles(CFG)[1] - 0.59, abs=0.02)
    assert ev.max_delta == pytest.approx(0.0, abs=1e-9)


def test_body_yaw_lags_tangent_and_rear_axle_cuts_inside():
    seg = QuinticSegment.connect(0.0, (0, 0, 0), 5.0, (1.0, 0, 0))
    tr = trace_path(OffsetPath([seg]), 0.0, 8.0, 0.0, CFG.vehicle.wheelbase, 0.05)
    mid = np.searchsorted(tr.s, 2.5)
    assert 0.0 < tr.psi[mid] < tr.theta[mid]                     # yaw lags the path tangent
    rear_d = tr.d - CFG.vehicle.wheelbase * np.sin(tr.psi)
    rear_s = tr.s - CFG.vehicle.wheelbase * np.cos(tr.psi)
    path_d_at_rear = seg.eval(rear_s)[0]
    # Left lane change: the rear axle runs to the RIGHT of (inside) the front-axle path.
    assert (rear_d - path_d_at_rear)[mid:].min() < -0.03
    assert tr.psi[-1] == pytest.approx(0.0, abs=0.05)             # converges once straight again


def test_tangent_yaw_assumption_would_misplace_rear_tyre():
    """Regression for the modelling choice: the two yaw models disagree where it matters."""
    seg = QuinticSegment.connect(0.0, (0, 0, 0), 4.0, (0.7, 0, 0))
    tr = trace_path(OffsetPath([seg]), 0.0, 6.0, 0.0, CFG.vehicle.wheelbase, 0.05)
    i = int(np.argmax(np.abs(tr.theta - tr.psi)))
    rear_true = tr.d[i] - CFG.front_axle_to_rear * math.sin(tr.psi[i])
    rear_tangent = tr.d[i] - CFG.front_axle_to_rear * math.sin(tr.theta[i])
    assert abs(rear_true - rear_tangent) > 0.10


def test_steering_demand_matches_curvature_for_gentle_paths():
    # Quasi-static limit (maneuver >> wheelbase): delta -> atan(wheelbase * kappa). For the
    # 4-6 m maneuvers of this mission the real demand is LOWER, because the yaw lag
    # low-passes the front-axle path; that is why the ODE is used instead of the closed form.
    D, L = 2.0, 60.0
    seg = QuinticSegment.connect(0.0, (0, 0, 0), L, (D, 0, 0))
    ev = evaluate(OffsetPath([seg]), 0.0, 0.0, np.zeros((0, 2)), CFG)
    kappa_max = 10 / math.sqrt(3) * D / L ** 2
    assert ev.max_delta == pytest.approx(math.atan(CFG.vehicle.wheelbase * kappa_max), rel=0.03)
    assert ev.clearance_circle == math.inf


def test_clearance_negative_when_driving_through_cone():
    circle, rect = clearance_at_pose(10.0, 0.0, 0.0, np.array([[10.0, 0.3]]), CFG)
    assert rect.min() < 0 and circle.min() < 0
