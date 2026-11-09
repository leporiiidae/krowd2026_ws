import math
import os
import struct

import pytest
import yaml

from cone_avoidance import can_node as codec
from cone_avoidance.mission_node import OffsetPath, ReferenceLine
from cone_avoidance.params import DEFAULTS, Config
from cone_avoidance.control_node import StanleyController

REF = ReferenceLine(0.0, 0.0, 0.0)


def test_stanley_steers_back_toward_the_path():
    cfg = Config({"control": {"latency_compensation": 0.0}})
    left = StanleyController(cfg).step(REF, OffsetPath(), 5.0, +0.3, 0.0, 1.0, 1.0)
    right = StanleyController(cfg).step(REF, OffsetPath(), 5.0, -0.3, 0.0, 1.0, 1.0)
    assert left.e_y == pytest.approx(0.3) and left.delta < 0      # left of path -> steer RIGHT
    assert right.delta > 0
    yawed = StanleyController(cfg).step(REF, OffsetPath(), 5.0, 0.0, -0.1, 1.0, 1.0)
    assert yawed.theta_e == pytest.approx(0.1) and yawed.delta > 0  # nose right -> steer LEFT


def test_stanley_is_bounded_at_standstill_and_rate_limited():
    cfg = Config()
    ctrl = StanleyController(cfg)
    out = ctrl.step(REF, OffsetPath(), 0.0, 0.5, 0.0, 0.0, 0.02)
    assert math.isfinite(out.delta_raw) and abs(out.delta_raw) < math.pi / 2
    assert abs(out.delta) <= cfg.max_steer_rate * 0.02 + 1e-12
    for _ in range(200):
        out = ctrl.step(REF, OffsetPath(), 0.0, 5.0, 0.0, 0.0, 0.02)
    assert abs(out.delta) == pytest.approx(cfg.max_wheel_steer)


def test_latency_prediction_moves_the_evaluation_point_forward():
    cfg = Config({"control": {"latency_compensation": 0.2}})
    out = StanleyController(cfg).step(REF, OffsetPath(), 5.0, 0.0, 0.0, 1.5, 0.02)
    assert out.s == pytest.approx(5.3)


def test_impossible_wheel_angle_is_rejected_not_silently_inverted():
    # tan() flips sign past 90 deg, so a bogus limit would invert every steering bound.
    with pytest.raises(ValueError, match="max_wheel_steer_deg"):
        Config({"vehicle": {"max_wheel_steer_deg": 100.0}})
    assert Config({"vehicle": {"max_wheel_steer_deg": 35.0}}).kappa_limit > 0


def test_usable_steer_is_the_tighter_of_stop_and_soft_limit():
    """0x120 carries the output shaft angle, so the firmware limit caps the WHEEL angle."""
    cfg = Config()                                   # ratio 6.8, limit 2.5 -> 21.1 deg > stop 21
    assert cfg.steer_limited_by == "mechanical"
    assert math.degrees(cfg.max_wheel_steer) == pytest.approx(21.0)
    lowered = Config({"steering": {"output_limit_rad": 1.7453}})   # the old firmware limit
    assert lowered.steer_limited_by == "output_limit"
    assert math.degrees(lowered.max_wheel_steer) == pytest.approx(math.degrees(1.7453 / 6.8), abs=0.01)
    assert cfg.kappa_limit > lowered.kappa_limit > 0


def test_steer_frame_carries_output_shaft_angle():
    data, out_rad, raw = codec.encode_steer(0.1, 4.0, 1.7453, 100)
    assert out_rad == pytest.approx(0.4) and raw == 40          # +0.1 rad wheel, LEFT positive
    assert data == struct.pack("<h", 40) + bytes(6)
    wheel, out_fb, _ = codec.decode_steer(data, 4.0, 100)
    assert wheel == pytest.approx(0.1) and out_fb == pytest.approx(0.4)
    _, clipped, raw = codec.encode_steer(-1.0, 4.0, 1.7453, 100)
    assert clipped == pytest.approx(-1.7453) and raw == -175


def test_quantisation_at_scale_100_is_small_at_the_wheel():
    """0.01 rad on the output shaft is only 0.084 deg of wheel angle at ratio 6.8."""
    cfg = Config()
    st = cfg.steering
    worst = 0.0
    steps = int(math.degrees(cfg.max_wheel_steer) * 10)
    for tenth_deg in range(-steps, steps + 1):        # the whole usable steering range
        wheel = math.radians(tenth_deg / 10.0)
        data, _, _ = codec.encode_steer(wheel, st.output_ratio, st.output_limit_rad, st.angle_scale)
        back, _, _ = codec.decode_steer(data, st.output_ratio, st.angle_scale)
        worst = max(worst, abs(back - wheel))
    assert math.degrees(worst) < 0.05


def test_speed_frame_is_kmh_times_100():
    data = codec.encode_speed(1.0, 100.0)
    assert struct.unpack("<h", data[:2])[0] == 360
    assert codec.decode_speed(data, 100.0) == pytest.approx(1.0)
    assert len(data) == 8 and codec.encode_zero_request()[0] == 1


def test_yaml_and_defaults_describe_the_same_parameters():
    path = os.path.join(os.path.dirname(__file__), "..", "config", "cone_avoidance.yaml")
    with open(path, encoding="utf-8") as handle:
        params = yaml.safe_load(handle)["/**"]["ros__parameters"]

    def keys(d, prefix=""):
        out = set()
        for k, v in d.items():
            out |= keys(v, prefix + k + ".") if isinstance(v, dict) else {prefix + k}
        return out

    assert keys(params) == keys(DEFAULTS)
    cfg = Config.from_yaml(path)
    assert cfg.collision_width == pytest.approx(1.18)
    assert cfg.steering.output_ratio == pytest.approx(6.8)   # measured 2026-09-19


def test_command_topics_carry_output_shaft_angle_by_default():
    """/target_angle goes straight onto 0x120, which carries the OUTPUT SHAFT angle."""
    cfg = Config()
    assert cfg.topics.steer_cmd == "/target_angle" and cfg.topics.speed_cmd == "/target_speed"
    angle, speed = codec.to_cmd_units(math.radians(10.0), 1.0, cfg)
    assert angle == pytest.approx(math.radians(10.0) * cfg.steering.output_ratio)
    assert speed == pytest.approx(3.6)                      # km/h by default
    wheel, mps = codec.from_cmd_units(angle, speed, cfg)    # what the bridge reads back
    assert math.degrees(wheel) == pytest.approx(10.0) and mps == pytest.approx(1.0)


def test_command_units_are_configurable_and_round_trip():
    for steer, speed_unit in (("wheel", "mps"), ("output_shaft", "kmh"), ("wheel", "kmh")):
        cfg = Config({"control": {"steer_cmd_unit": steer, "speed_cmd_unit": speed_unit}})
        angle, speed = codec.to_cmd_units(0.2, 1.5, cfg)
        if speed_unit == "kmh":
            assert speed == pytest.approx(5.4)
        wheel, mps = codec.from_cmd_units(angle, speed, cfg)
        assert wheel == pytest.approx(0.2) and mps == pytest.approx(1.5)


def test_published_steering_never_exceeds_the_soft_limit():
    cfg = Config()
    angle, _ = codec.to_cmd_units(1.0, 0.0, cfg)            # far past the limit
    assert angle == pytest.approx(cfg.steering.output_limit_rad)
    angle, _ = codec.to_cmd_units(1.0, 0.0, Config({"control": {"steer_cmd_unit": "wheel"}}))
    assert angle == pytest.approx(cfg.max_wheel_steer)
