import math

import numpy as np
import pytest

from cone_avoidance.geometry import (QUINTIC_MAX_D1, QUINTIC_MAX_D2, QUINTIC_MAX_D3, OffsetPath,
                                     QuinticSegment, ReferenceLine, wrap)


def test_frenet_left_is_positive_and_roundtrips():
    ref = ReferenceLine(10.0, -5.0, math.radians(90.0))   # heading north
    s, d = ref.to_frenet(9.0, -2.0)                        # 3 m ahead, 1 m to the WEST = left
    assert s == pytest.approx(3.0) and d == pytest.approx(1.0)
    assert ref.to_global(s, d) == pytest.approx((9.0, -2.0))
    arr = ref.to_frenet_array(np.array([[9.0, -2.0], [11.0, -5.0]]))
    assert arr[1] == pytest.approx([0.0, -1.0])


def test_wrap():
    assert wrap(3 * math.pi) == pytest.approx(math.pi)
    assert wrap(-math.pi / 2 - 2 * math.pi) == pytest.approx(-math.pi / 2)


def test_quintic_meets_general_boundary_conditions():
    seg = QuinticSegment.connect(2.0, (0.1, 0.05, -0.02), 9.0, (0.6, -0.03, 0.01))
    assert [float(v) for v in seg.eval(2.0)] == pytest.approx([0.1, 0.05, -0.02])
    assert [float(v) for v in seg.eval(9.0)] == pytest.approx([0.6, -0.03, 0.01])
    h = 1e-5   # analytic derivatives agree with finite differences
    d_lo, d_hi = float(seg.eval(5.0 - h)[0]), float(seg.eval(5.0 + h)[0])
    assert float(seg.eval(5.0)[1]) == pytest.approx((d_hi - d_lo) / (2 * h), abs=1e-6)


def test_rest_to_rest_maxima_match_spec_constants():
    D, L = 0.7, 5.0
    seg = QuinticSegment.connect(0.0, (0, 0, 0), L, (D, 0, 0))
    s = np.linspace(0, L, 20001)
    _, d1, d2 = seg.eval(s)
    d3 = np.gradient(d2, s)
    assert np.abs(d1).max() == pytest.approx(QUINTIC_MAX_D1 * D / L, rel=1e-3)
    assert np.abs(d2).max() == pytest.approx(QUINTIC_MAX_D2 * D / L ** 2, rel=1e-3)
    assert np.abs(d3).max() == pytest.approx(QUINTIC_MAX_D3 * D / L ** 3, rel=1e-2)


def test_replan_splice_is_c2_and_keeps_driven_part():
    path = OffsetPath([QuinticSegment.connect(0.0, (0, 0, 0), 8.0, (0.5, 0, 0))])
    cut = 3.0
    before = path.eval(cut)
    new = path.replace_from(cut, [QuinticSegment.connect(cut, before, 8.0, (0.62, 0, 0))])
    assert new.eval(cut - 1e-9) == pytest.approx(before, abs=1e-6)
    assert new.eval(cut + 1e-9) == pytest.approx(before, abs=1e-6)
    assert new.eval(1.7) == pytest.approx(path.eval(1.7))       # already-driven part unchanged
    assert new.eval(8.0)[0] == pytest.approx(0.62)
    assert new.eval(20.0) == pytest.approx((0.62, 0.0, 0.0))    # holds after the maneuver


def test_path_serialisation_roundtrip():
    path = OffsetPath([QuinticSegment.connect(1.0, (0, 0, 0), 6.0, (0.4, 0, 0)),
                       QuinticSegment.connect(7.0, (0.4, 0, 0), 11.0, (-0.4, 0, 0))])
    again = OffsetPath.from_list(path.to_list())
    for s in (0.0, 3.3, 6.5, 9.1, 15.0):
        assert again.eval(s) == pytest.approx(path.eval(s))
    d, _, _ = path.eval_array(np.array([0.0, 6.5, 15.0]))
    assert d == pytest.approx([0.0, 0.4, -0.4])
