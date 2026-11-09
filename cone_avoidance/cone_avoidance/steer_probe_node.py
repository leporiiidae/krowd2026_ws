"""Measure the steering: step response, rate limit and lag. Wheels off the ground.

    ros2 run cone_avoidance steer_probe
    ros2 run cone_avoidance steer_probe --ros-args -p amplitudes_deg:="[3.0, 6.0, 12.0, 18.0]"

Publishes only the steering command (never a speed), steps it up and down, and reports
what the wheel actually did — the two numbers the planner needs and that are still
guesses in the YAML: ``steering.max_rate_deg_s`` and ``control.latency_compensation``.
"""

from __future__ import annotations

import math
from typing import List, Optional

from rclpy.node import Node
from std_msgs.msg import Float32

from .can_codec import to_cmd_units
from .ros_common import declare_config, run_node


class Step:
    def __init__(self, target: float, start: float, begin_angle: float):
        self.target = target          # wheel rad
        self.start = start
        self.begin = begin_angle
        self.samples: List[tuple] = []   # (t, wheel rad)

    def slew_rate(self) -> float:
        """Slope over the middle of the travel.

        Sample-to-sample differences only resolve one CAN count per control period
        (0.084 deg / 20 ms = 4.2 deg/s here), so a per-sample maximum lands on a
        multiple of that. Fitting the 20-80% segment removes the quantisation.
        """
        travel = self.target - self.begin
        if abs(travel) < 1e-6 or len(self.samples) < 4:
            return 0.0
        band = [(t, a) for t, a in self.samples if 0.2 <= (a - self.begin) / travel <= 0.8]
        if len(band) < 3:
            return 0.0
        span_t = band[-1][0] - band[0][0]
        return abs(band[-1][1] - band[0][1]) / span_t if span_t > 1e-6 else 0.0

    def report(self, logger) -> None:
        if len(self.samples) < 5:
            logger.warn(f"step to {math.degrees(self.target):+.1f} deg: no feedback")
            return
        t0 = self.start
        travel = self.target - self.begin
        if abs(travel) < math.radians(0.5):
            return
        reached = lambda frac: next(
            (t - t0 for t, a in self.samples
             if (a - self.begin) / travel >= frac), None)
        t10, t63, t90 = reached(0.1), reached(0.63), reached(0.9)
        rate = self.slew_rate()
        final = self.samples[-1][1]
        logger.info(
            f"step {math.degrees(self.begin):+6.1f} -> {math.degrees(self.target):+6.1f} deg | "
            f"dead time {t10 if t10 is None else round(t10, 2)} s | "
            f"63% at {t63 if t63 is None else round(t63, 2)} s | "
            f"90% at {t90 if t90 is None else round(t90, 2)} s | "
            f"peak rate {math.degrees(rate):5.1f} deg/s | "
            f"settled {math.degrees(final):+6.1f} deg "
            f"({100 * (final - self.begin) / travel:.0f}% of the step)")


class SteerProbeNode(Node):
    def __init__(self):
        super().__init__("cone_steer_probe")
        self.cfg = declare_config(self)
        self.declare_parameter("amplitudes_deg", [3.0, 6.0, 12.0, 18.0])
        self.declare_parameter("dwell", 2.0)
        amps = list(self.get_parameter("amplitudes_deg").value)
        self.dwell = float(self.get_parameter("dwell").value)
        # up, back to centre, down, back to centre, ... so every step is measured twice
        self.sequence = [0.0]
        for a in amps:
            self.sequence += [math.radians(a), 0.0, -math.radians(a), 0.0]

        self.index = -1
        self.step: Optional[Step] = None
        self.done: List[Step] = []
        self.feedback: Optional[float] = None
        self.cmd = 0.0
        self.finished = False

        t = self.cfg.topics
        self.create_subscription(Float32, t.steer_feedback, self.on_feedback, 10)
        self.cmd_pub = self.create_publisher(Float32, t.steer_cmd, 10)
        self.create_timer(0.02, self.publish_cmd)          # 50 Hz, like the controller
        self.create_timer(self.dwell, self.next_step)
        self.get_logger().warn("STEERING PROBE: wheels off the ground. No speed command is sent, "
                               "but the drive firmware may still be armed.")

    def on_feedback(self, msg: Float32) -> None:
        self.feedback = float(msg.data)
        if self.step is not None:
            self.step.samples.append((self.now(), self.feedback))

    def now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_cmd(self) -> None:
        angle, _ = to_cmd_units(self.cmd, 0.0, self.cfg)
        self.cmd_pub.publish(Float32(data=float(angle)))

    def next_step(self) -> None:
        if self.step is not None:
            self.step.report(self.get_logger())
            self.done.append(self.step)
            self.step = None
        self.index += 1
        if self.index >= len(self.sequence):
            if not self.finished:                      # the timer keeps firing; report once
                self.finished = True
                self.summary()
            self.cmd = 0.0
            return
        begin = self.feedback if self.feedback is not None else self.cmd
        self.cmd = self.sequence[self.index]
        self.step = Step(self.cmd, self.now(), begin)

    def summary(self) -> None:
        """Fit the actuator: rate = slope * error, saturating at max_rate.

        The steering firmware sets the step frequency from the position error, so a step
        smaller than the saturation error never reaches full speed and its response is a
        first-order lag with tau = 1 / slope. That tau, not the dead time, is what the
        controller has to lead by.
        """
        rates, dead, tau = [], [], []
        slopes = []
        for step in self.done:
            if len(step.samples) < 5:
                continue
            travel = step.target - step.begin
            if abs(travel) < math.radians(2.0):
                continue
            peak = step.slew_rate()
            rates.append(peak)
            if peak < 0.92 * max(r for r in rates):      # this step never saturated
                slopes.append(peak / abs(travel))
            t10 = next((t - step.start for t, a in step.samples
                        if (a - step.begin) / travel >= 0.1), None)
            if t10 is not None:
                dead.append(t10)
            t63 = next((t - step.start for t, a in step.samples
                        if (a - step.begin) / travel >= 0.63), None)
            if t63 is not None:
                tau.append(t63)
        if not rates:
            self.get_logger().error("no usable steps — was /cone_avoidance/steer_feedback published?")
            return
        rate = math.degrees(max(rates))
        dead_time = min(dead) if dead else float("nan")   # the smallest step shows the pure delay
        slope = max(slopes) if slopes else float("nan")
        # The controller must lead by the WHOLE lag, not just the dead time: the wheel
        # keeps moving for another time constant afterwards. Time to 63% covers both,
        # and simulating this vehicle shows why it matters — with only the dead time the
        # cross-track error is 5x larger and the loop oscillates at higher gain.
        if slope and slope == slope and slope > 0:
            time_constant = 1.0 / slope
        else:   # every step saturated: fall back to the smallest step's 63% time
            time_constant = (min(tau) - dead_time) if tau else 0.0
        effective = dead_time + max(time_constant, 0.0)
        self.get_logger().info("=" * 62)
        self.get_logger().info(f"  steering.max_rate_deg_s     : {rate:.0f}   (currently "
                               f"{self.cfg.steering.max_rate_deg_s:.0f})")
        self.get_logger().info(f"  control.latency_compensation: {effective:.2f}   (currently "
                               f"{self.cfg.control.latency_compensation:.2f})")
        self.get_logger().info(f"     = dead time {dead_time:.2f} s + steering time constant "
                               f"{time_constant:.2f} s")
        sat_deg = math.degrees(rate_sat) if (rate_sat := math.radians(rate) / max(slope, 1e-6)) else 0.0
        if slope and slope == slope:
            self.get_logger().info(f"     the wheel moves at {slope:.1f} x error, saturating at "
                                   f"{sat_deg:.1f} deg of error")
            self.get_logger().info("     -> firmware MAX_RATE_ERROR_RAD sets this; lowering it cuts "
                                   "the time constant in proportion")
        self.get_logger().info("  put these in config/cone_avoidance.yaml — the planner sizes every")
        self.get_logger().info("  maneuver and picks the speed from them.")
        self.get_logger().info("=" * 62)


def main(args=None):
    run_node(SteerProbeNode)


if __name__ == "__main__":
    main()
