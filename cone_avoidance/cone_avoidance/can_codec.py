"""CAN frame encoding/decoding (500 kbps, standard frames, int16 little-endian)."""

from __future__ import annotations

import struct

ID_SPEED_CMD = 0x110
ID_STEER_CMD = 0x120
ID_ZERO_REQ = 0x130
ID_WHEEL_SPEED = 0x200
ID_STEER_FB = 0x201
ID_ZERO_DONE = 0x202

INT16_MIN, INT16_MAX = -32768, 32767
MPS_TO_KMH = 3.6


def _int16(value: float) -> int:
    return max(INT16_MIN, min(INT16_MAX, int(round(value))))


def encode_speed(v_mps: float, speed_scale: float) -> bytes:
    return struct.pack("<h", _int16(v_mps * MPS_TO_KMH * speed_scale)) + bytes(6)


def decode_speed(data: bytes, speed_scale: float) -> float:
    return struct.unpack("<h", bytes(data[:2]))[0] / speed_scale / MPS_TO_KMH


def wheel_to_output(wheel_rad: float, output_ratio: float, output_limit: float) -> float:
    """0x120 carries the steering OUTPUT SHAFT angle, not the road-wheel angle."""
    out = wheel_rad * output_ratio
    return max(-output_limit, min(output_limit, out))


def encode_steer(wheel_rad: float, output_ratio: float, output_limit: float,
                 angle_scale: float):
    out = wheel_to_output(wheel_rad, output_ratio, output_limit)
    raw = _int16(out * angle_scale)
    return struct.pack("<h", raw) + bytes(6), out, raw


def decode_steer(data: bytes, output_ratio: float, angle_scale: float):
    """Returns (wheel_rad, output_rad, raw)."""
    raw = struct.unpack("<h", bytes(data[:2]))[0]
    out = raw / angle_scale
    return out / output_ratio, out, raw


def encode_zero_request() -> bytes:
    return bytes([1]) + bytes(7)


def is_zero_done(data: bytes) -> bool:
    return len(data) >= 1 and data[0] == 1


def to_cmd_units(wheel_rad: float, speed_mps: float, cfg):
    """Convert the controller's (wheel angle, m/s) into the units the command topics carry.

    The steering topic may be read by this package's can_bridge or by an external
    bridge that puts the value straight on 0x120 (which carries the OUTPUT SHAFT
    angle), so the unit is a parameter and both ends read the same one.
    """
    st = cfg.steering
    if cfg.control.steer_cmd_unit == "wheel":
        angle = max(-cfg.max_wheel_steer, min(cfg.max_wheel_steer, wheel_rad))
    else:
        angle = wheel_to_output(wheel_rad, st.output_ratio, st.output_limit_rad)
    speed = speed_mps * MPS_TO_KMH if cfg.control.speed_cmd_unit == "kmh" else speed_mps
    return angle, speed


def from_cmd_units(angle: float, speed: float, cfg):
    """Inverse of :func:`to_cmd_units` — what the CAN bridge received, as (wheel rad, m/s)."""
    st = cfg.steering
    wheel = angle if cfg.control.steer_cmd_unit == "wheel" else angle / st.output_ratio
    speed_mps = speed / MPS_TO_KMH if cfg.control.speed_cmd_unit == "kmh" else speed
    return wheel, speed_mps
