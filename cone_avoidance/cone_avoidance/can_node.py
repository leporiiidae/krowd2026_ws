"""SocketCAN bridge: commands out (0x110/0x120/0x130), feedback in (0x200/0x201/0x202)."""

from __future__ import annotations

import json
import math

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float32, String

from . import can_codec as codec
from .can_codec import from_cmd_units
from .ros_common import run_node, declare_config


class CanNode(Node):
    def __init__(self):
        super().__init__("cone_can")
        self.cfg = declare_config(self)
        st = self.cfg.steering
        if not st.output_ratio or st.output_ratio <= 0.0:
            # A wrong ratio silently under-steers into the cones; an unset one must not run.
            raise RuntimeError(
                "steering.output_ratio is not set. Measure it (spec 15-2: sweep lock to lock, "
                "record 0x201 vs protractor wheel angle) and put it in the YAML.")
        try:
            import can
        except ImportError as error:
            raise RuntimeError("python-can is required: pip install python-can") from error
        self._can = can
        # The interface itself is brought up outside this node (ip link set <if> up);
        # here it is only opened for send/receive.
        self.bus = can.interface.Bus(channel=self.cfg.can.interface, interface="socketcan")

        t = self.cfg.topics
        self.steer_cmd = 0.0
        self.speed_cmd = 0.0
        self.cmd_time = -math.inf
        self.rx_time = {codec.ID_WHEEL_SPEED: -math.inf, codec.ID_STEER_FB: -math.inf}
        self.zero_done = False
        self.zero_req_time = -math.inf
        self.last_out = 0.0
        self.fb_out = math.nan
        self.mismatch_count = 0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(Float32, t.steer_cmd, self.on_steer, 10)
        self.create_subscription(Float32, t.speed_cmd, self.on_speed, 10)
        self.create_subscription(Bool, t.zero_request, self.on_zero_request, 10)
        self.fb_pub = self.create_publisher(Float32, t.steer_feedback, 10)
        self.speed_pub = self.create_publisher(Float32, t.wheel_speed, 10)
        self.zero_pub = self.create_publisher(Bool, t.zero_done, latched)
        self.status_pub = self.create_publisher(String, t.can_status, 10)
        self.create_timer(1.0 / max(float(self.cfg.can.tx_rate_hz), 1.0), self.tick)
        self.get_logger().info(
            f"CAN opened on {self.cfg.can.interface} "
            f"(zero request on start: {'on' if self.cfg.can.auto_zero_request else 'off'}): "
            f"output_ratio={st.output_ratio}, "
            f"angle_scale={st.angle_scale} (must equal firmware ANGLE_SCALE), "
            f"usable wheel steer={math.degrees(self.cfg.max_wheel_steer):.1f} deg "
            f"(limited by {self.cfg.steer_limited_by})")

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def on_steer(self, msg):
        # The topic carries control.steer_cmd_unit; internally everything is wheel angle.
        self.steer_cmd = from_cmd_units(float(msg.data), 0.0, self.cfg)[0]
        self.cmd_time = self.now()

    def on_speed(self, msg):
        self.speed_cmd = from_cmd_units(0.0, float(msg.data), self.cfg)[1]

    def on_zero_request(self, msg):
        if msg.data:
            self.zero_done = False
            self._send(codec.ID_ZERO_REQ, codec.encode_zero_request())

    def _send(self, can_id, data):
        try:
            self.bus.send(self._can.Message(arbitration_id=can_id, data=data, is_extended_id=False))
            return True
        except self._can.CanError as error:
            self.get_logger().error(f"CAN send failed: {error}", throttle_duration_sec=1.0)
            return False

    def _receive(self, now):
        st = self.cfg.steering
        for _ in range(100):
            frame = self.bus.recv(timeout=0.0)
            if frame is None:
                return
            if frame.is_extended_id or len(frame.data) < 1:
                continue
            if frame.arbitration_id == codec.ID_WHEEL_SPEED and len(frame.data) >= 2:
                self.rx_time[codec.ID_WHEEL_SPEED] = now
                self.speed_pub.publish(Float32(data=codec.decode_speed(frame.data, self.cfg.can.speed_scale)))
            elif frame.arbitration_id == codec.ID_STEER_FB and len(frame.data) >= 2:
                self.rx_time[codec.ID_STEER_FB] = now
                wheel, self.fb_out, _ = codec.decode_steer(frame.data, st.output_ratio, st.angle_scale)
                self.fb_pub.publish(Float32(data=wheel))
            elif frame.arbitration_id == codec.ID_ZERO_DONE and codec.is_zero_done(frame.data):
                if not self.zero_done:
                    self.get_logger().info("steering zero done (0x202)")
                self.zero_done = True
                self.zero_pub.publish(Bool(data=True))

    def _scale_mismatch(self) -> bool:
        """Feedback settling at ~10x or ~1/10 of the command = PC/firmware ANGLE_SCALE differ."""
        if not math.isfinite(self.fb_out) or abs(self.last_out) < 0.1:
            self.mismatch_count = 0
            return False
        ratio = self.fb_out / self.last_out
        off = 0.07 <= abs(ratio) <= 0.14 or 7.0 <= abs(ratio) <= 14.0
        self.mismatch_count = self.mismatch_count + 1 if off else 0
        return self.mismatch_count > int(self.cfg.can.tx_rate_hz)  # > 1 s

    def tick(self):
        cfg = self.cfg
        st = cfg.steering
        now = self.now()
        self._receive(now)

        if cfg.can.auto_zero_request and not self.zero_done and now - self.zero_req_time >= 1.0:
            self.zero_req_time = now
            self._send(codec.ID_ZERO_REQ, codec.encode_zero_request())

        cmd_lost = now - self.cmd_time > cfg.can.cmd_timeout
        speed = self.speed_cmd
        if cmd_lost and cfg.can.on_cmd_timeout == "stop":
            speed = 0.0  # the controller process is gone: nobody is steering
        steer_data, self.last_out, _ = codec.encode_steer(
            self.steer_cmd, st.output_ratio, st.output_limit_rad, st.angle_scale)
        self._send(codec.ID_SPEED_CMD, codec.encode_speed(speed, cfg.can.speed_scale))
        self._send(codec.ID_STEER_CMD, steer_data)

        rx_lost = {hex(k): now - v > cfg.can.rx_timeout for k, v in self.rx_time.items()}
        mismatch = self._scale_mismatch()
        if mismatch:
            self.get_logger().error("0x201 is ~10x off the command: ANGLE_SCALE differs between "
                                    "PC and firmware", throttle_duration_sec=2.0)
        moving = abs(self.speed_cmd) > 0.0
        self.status_pub.publish(String(data=json.dumps({
            "fault": bool(moving and (any(rx_lost.values()) or mismatch)),
            "rx_lost": rx_lost, "cmd_lost": bool(cmd_lost), "zero_done": self.zero_done,
            "angle_scale_mismatch": bool(mismatch)})))

    def destroy_node(self):
        try:
            self.bus.shutdown()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    run_node(CanNode)


if __name__ == "__main__":
    main()
