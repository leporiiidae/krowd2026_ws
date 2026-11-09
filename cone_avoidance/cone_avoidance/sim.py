"""Kinematic simulator + Monte Carlo for the cone-avoidance stack.

Runs the SAME Mission / Planner / Stanley / ConeTracker code as the ROS nodes
against a bicycle model with
  * steering actuator: pure delay + error-proportional rate with saturation
    (the stepper firmware moves slowly on small errors),
  * localisation noise, cone-estimate noise and detection drop-outs,
  * initial lateral / heading error,
so snaking and tracking lag show up before field testing.

    python3 -m cone_avoidance.sim --seeds 40
    python3 -m cone_avoidance.sim --seeds 40 --gap 4 --set speed.target=1.5
    python3 -m cone_avoidance.sim --seed 3 --plot run.png --csv run.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import Pool
from typing import List, Optional

import numpy as np
import yaml

from .mission_node import DONE, Mission, clearance_at_pose
from .params import Config, deep_merge, unflatten
from .control_node import StanleyController, from_cmd_units
from .perception_node import ConeTracker, RowFilter


@dataclass
class RunResult:
    seed: int
    min_clearance: float          # exact rectangle vs true cones
    min_clearance_circle: float
    collided: bool
    done: bool
    max_abs_ey: float
    max_lane_excursion: float     # >0: body beyond the solid line, or beyond the centre line where not allowed
    max_center_cross: float       # >0: body beyond the centre line where the rules allow it
    min_speed: float
    max_track_err_deg: float
    duration: float
    flags: List[str] = field(default_factory=list)
    states: List[str] = field(default_factory=list)
    log: Optional[list] = None


class Simulator:
    def __init__(self, cfg: Config, seed: int = 0, keep_log: bool = False):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.seed = seed
        self.keep_log = keep_log
        self.cones = np.asarray(cfg.sim.cones, dtype=float).reshape(-1, 2)

    # The world frame IS the road frame: x along the road, y left, lane centre y = 0.
    def run(self) -> RunResult:
        cfg, sim, rng = self.cfg, self.cfg.sim, self.rng
        wb = cfg.vehicle.wheelbase
        dt = sim.dt

        yaw = math.radians(rng.normal(0.0, sim.init_yaw_std_deg))
        y_f0 = rng.normal(0.0, sim.init_d_std)
        x_r, y_r = -wb * math.cos(yaw), y_f0 - wb * math.sin(yaw)
        v = 0.0
        delta = 0.0

        mission = Mission(cfg)
        tracker = ConeTracker(cfg)
        row_filter = RowFilter(cfg)
        stanley = StanleyController(cfg)
        mission.on_zero_done()
        mission.on_start()

        steer_queue = deque([0.0] * max(1, int(round(sim.steer_delay / dt))))
        # What the steering CAN do, which is not always what the planner was told it can
        # do: a mismatch here is exactly the field failure (plans too fast to follow).
        plant_rate = (math.radians(sim.actual_max_rate_deg_s) if sim.actual_max_rate_deg_s > 0
                      else cfg.max_steer_rate)
        k_act = plant_rate / max(sim.steer_err_sat_rad, 1e-3)
        every = lambda hz: max(1, int(round(1.0 / (hz * dt))))
        n_plan, n_lidar, n_yolo = every(10.0), every(10.0), every(sim.yolo_rate_hz)
        n_odom = every(sim.odom_rate_hz)

        meas = (0.0, y_f0, yaw, 0.0)
        v_cmd, delta_cmd = 0.0, 0.0
        min_rect = min_circ = math.inf
        max_ey = max_exc = max_trk = max_cross = 0.0
        # Rules: the centre line (left) may be crossed only while avoiding the listed
        # groups. Ground truth: group n is "being avoided" once group n-1 is abeam.
        # Ground truth: group n is "being avoided" once the body is past group n-1.
        group_s = sorted({round(float(v), 1) for v in self.cones[:, 0]})
        cross_ok_from = min([group_s[i - 2] + cfg.cone_radius if i >= 2 else -math.inf
                             for i in map(int, cfg.road.center_cross_groups)
                             if i - 1 < len(group_s)] or [math.inf])
        min_speed = math.inf
        states, log = [], []
        t, step = 0.0, 0
        half_w = cfg.collision_width / 2.0
        corners_x = np.array([-cfg.vehicle.rear_overhang, cfg.rear_axle_to_front])

        while t < 120.0:
            x_f, y_f = x_r + wb * math.cos(yaw), y_r + wb * math.sin(yaw)
            if step % n_odom == 0:
                meas = (x_f + rng.normal(0, sim.pos_noise_std), y_f + rng.normal(0, sim.pos_noise_std),
                        yaw + math.radians(rng.normal(0, sim.yaw_noise_std_deg)), v)

            if step % n_lidar == 0:
                obs = self._lidar(x_r, y_r, yaw, meas)
                tracker.update(obs)
                mission.on_lidar_frame(obs, t)
            if step % n_yolo == 0:
                mission.on_yolo(self._yolo(x_f, y_f, yaw), t)
            if step % n_plan == 0:
                tracks = [(tr.track_id, tr.x, tr.y) for tr in row_filter.update(tracker.confirmed(), meas[2], meas[:2])]
                base = (meas[0] - wb * math.cos(meas[2]), meas[1] - wb * math.sin(meas[2]))
                mission.update(t, meas[0], meas[1], meas[2], meas[3], tracks, base)
                v_cmd = mission.speed_command()
                if not states or states[-1] != mission.state:
                    states.append(mission.state)

            out = None
            if mission.ref is not None:
                out = stanley.step(mission.ref, mission.path, meas[0], meas[1], meas[2], meas[3], dt)
                # The external CAN bridge is outside this package.  The plant receives
                # the controller's bounded road-wheel command directly.
                delta_cmd = out.delta

            # --- plant -----------------------------------------------------
            steer_queue.append(delta_cmd)
            target = steer_queue.popleft()
            rate = max(-plant_rate, min(plant_rate, k_act * (target - delta)))
            delta += rate * dt
            v += max(-sim.accel_limit * dt, min(sim.accel_limit * dt, v_cmd - v))
            x_r += v * math.cos(yaw) * dt
            y_r += v * math.sin(yaw) * dt
            yaw += v / wb * math.tan(delta) * dt
            t += dt
            step += 1

            # --- ground-truth scoring ---------------------------------------
            x_f, y_f = x_r + wb * math.cos(yaw), y_r + wb * math.sin(yaw)
            circ, rect = clearance_at_pose(x_f, y_f, yaw, self.cones, cfg)
            min_rect, min_circ = min(min_rect, float(rect.min())), min(min_circ, float(circ.min()))
            cx = np.concatenate([x_r + corners_x * math.cos(yaw) - w * math.sin(yaw) for w in (-half_w, half_w)])
            cy = np.concatenate([y_r + corners_x * math.sin(yaw) + w * math.cos(yaw) for w in (-half_w, half_w)])
            if mission.state != "IDLE" and x_f > 1.0:
                max_exc = max(max_exc, float(-cy.min()) - cfg.road.width / 2.0)
                over_left = cy - cfg.road.width / 2.0
                ok = cx >= cross_ok_from
                if ok.any():
                    max_cross = max(max_cross, float(over_left[ok].max()))
                if (~ok).any():
                    max_exc = max(max_exc, float(over_left[~ok].max()))
                min_speed = min(min_speed, v)
            if out is not None:
                max_ey = max(max_ey, abs(y_f - mission.path.eval(x_f)[0]) if cfg.reference.mode == "fixed"
                             else abs(out.e_y))
                max_trk = max(max_trk, abs(delta_cmd - delta))
            if self.keep_log:
                # The planned path lives in the mission's reference frame, which in
                # start_pose mode is rotated by the initial heading error: log it in world.
                path_xy = mission.ref.to_global(out.s, out.d_path) if out else (x_f, 0.0)
                log.append({
                    "path_x": path_xy[0], "path_y": path_xy[1],
                    "t": t, "x_f": x_f, "y_f": y_f, "yaw": yaw, "v": v, "v_cmd": v_cmd,
                    "delta_cmd": delta_cmd, "delta": delta, "state": mission.state,
                    "d_path": out.d_path if out else 0.0, "e_y": out.e_y if out else 0.0,
                    "theta_e": out.theta_e if out else 0.0,
                    "clr_rect": float(rect.min()), "clr_circle": float(circ.min()),
                    "x_r": x_r, "y_r": y_r,
                })
            if mission.state == DONE or x_f > sim.course_length:
                break

        return RunResult(self.seed, min_rect, min_circ, min_rect < 0.0, mission.state == DONE,
                         max_ey, max_exc, max_cross, min_speed, math.degrees(max_trk), t,
                         [k for k, val in mission.flags.items() if val], states,
                         log if self.keep_log else None)

    def _lidar(self, x_r, y_r, yaw, meas):
        """Noisy cone centres, measured in the TRUE vehicle frame, placed with the MEASURED pose."""
        cfg, sim, rng = self.cfg, self.cfg.sim, self.rng
        p = cfg.perception
        c, s = math.cos(yaw), math.sin(yaw)
        wb = cfg.vehicle.wheelbase
        out = []
        for cx, cy in self.cones:
            dx, dy = cx - x_r, cy - y_r
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            dist = math.hypot(lx, ly)
            if not (p.roi_x[0] <= lx <= min(p.roi_x[1], p.detect_range) and p.roi_y[0] <= ly <= p.roi_y[1]):
                continue
            frac = min(1.0, dist / p.detect_range)
            if rng.random() > sim.cone_detect_prob_near + frac * (sim.cone_detect_prob_far - sim.cone_detect_prob_near):
                continue
            std = sim.cone_noise_std_near + sim.cone_noise_std_per_m * dist
            lx += rng.normal(0, std)
            ly += rng.normal(0, std)
            cm, sm = math.cos(meas[2]), math.sin(meas[2])
            xr_m, yr_m = meas[0] - wb * cm, meas[1] - wb * sm
            out.append((xr_m + cm * lx - sm * ly, yr_m + sm * lx + cm * ly))
        return out

    def _yolo(self, x_f, y_f, yaw) -> int:
        sim = self.cfg.sim
        c, s = math.cos(yaw), math.sin(yaw)
        count = 0
        for cx, cy in self.cones:
            dx, dy = cx - x_f, cy - y_f
            lx, ly = c * dx + s * dy, -s * dx + c * dy
            if 0.3 < lx <= sim.yolo_range and abs(math.degrees(math.atan2(ly, lx))) <= sim.yolo_fov_deg / 2:
                count += 1
        return count


# ------------------------------------------------------------------ CLI
def _run_one(args):
    raw, seed = args
    return Simulator(Config(raw), seed).run()


def monte_carlo(cfg: Config, seeds: int, workers: Optional[int] = None) -> List[RunResult]:
    jobs = [(cfg.raw, seed) for seed in range(seeds)]
    if workers == 1:
        return [_run_one(j) for j in jobs]
    with Pool(workers) as pool:
        return pool.map(_run_one, jobs)


def summarize(results: List[RunResult]) -> str:
    clr = np.array([r.min_clearance for r in results])
    circ = np.array([r.min_clearance_circle for r in results])
    lines = [
        f"runs                 : {len(results)}",
        f"collisions           : {sum(r.collided for r in results)}",
        f"DONE reached         : {sum(r.done for r in results)}",
        f"cone clearance (rect): min {clr.min():+.3f}  mean {clr.mean():+.3f}  std {clr.std():.3f}  [m]",
        f"cone clearance (3circ): min {circ.min():+.3f}  mean {circ.mean():+.3f}",
        f"lane violation  max  : {max(r.max_lane_excursion for r in results):+.3f} m  (>0 = over solid line / centre line where forbidden)",
        f"centre-line cross max: {max(r.max_center_cross for r in results):+.3f} m  (allowed zone only)",
        f"cross-track err max  : {max(r.max_abs_ey for r in results):.3f} m",
        f"steer tracking err   : {max(r.max_track_err_deg for r in results):.2f} deg max",
        f"speed min / duration : {min(r.min_speed for r in results):.2f} m/s / {np.mean([r.duration for r in results]):.1f} s",
    ]
    flags = sorted({f for r in results for f in r.flags})
    if flags:
        lines.append("flags                : " + ", ".join(
            f"{f} x{sum(f in r.flags for r in results)}" for f in flags))
    worst = min(results, key=lambda r: r.min_clearance)
    lines.append(f"worst seed           : {worst.seed}  states={'>'.join(worst.states)}")
    return "\n".join(lines)


def _parse_value(text: str):
    return yaml.safe_load(text)


def build_config(path: Optional[str], sets: List[str], gap: Optional[float]) -> Config:
    overrides = unflatten({k: _parse_value(v) for k, v in (item.split("=", 1) for item in sets)})
    cfg = Config.from_yaml(path, overrides) if path else Config(overrides)
    if gap is not None:
        cones = np.asarray(cfg.sim.cones, dtype=float).reshape(-1, 2)
        first = cones[:, 0].min()
        cones[cones[:, 0] > first + cfg.perception.pair_group_window, 0] = first + gap
        cfg = Config(deep_merge(cfg.raw, {"sim": {"cones": cones.reshape(-1).tolist()}}))
    return cfg


def default_config_path() -> Optional[str]:
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "config", "cone_avoidance.yaml")
    if os.path.exists(here):
        return here
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("cone_avoidance"), "config", "cone_avoidance.yaml")
    except Exception:
        return None


def plot(result: RunResult, cfg: Config, path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = result.log
    x = [r["x_f"] for r in log]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    ax = axes[0]
    half = cfg.road.width / 2
    ax.axhline(half, color="k"); ax.axhline(-half, color="k"); ax.axhline(0, color="0.8", ls="--")
    for cx, cy in np.asarray(cfg.sim.cones).reshape(-1, 2):
        ax.add_patch(plt.Circle((cx, cy), cfg.cone_radius, color="tab:orange"))
    ax.plot([r["path_x"] for r in log], [r["path_y"] for r in log], "g--", label="path (front axle)")
    ax.plot(x, [r["y_f"] for r in log], "b", label="front axle")
    ax.plot([r["x_r"] for r in log], [r["y_r"] for r in log], "r", label="rear axle")
    for r in log[::25]:
        c, s = math.cos(r["yaw"]), math.sin(r["yaw"])
        xs = [-cfg.vehicle.rear_overhang, cfg.rear_axle_to_front]
        hw = cfg.collision_width / 2
        pts = [(xs[0], -hw), (xs[1], -hw), (xs[1], hw), (xs[0], hw), (xs[0], -hw)]
        ax.plot([r["x_r"] + c * px - s * py for px, py in pts],
                [r["y_r"] + s * px + c * py for px, py in pts], color="0.6", lw=0.6)
    ax.set_aspect("equal"); ax.set_ylabel("y [m]"); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"seed {result.seed}: min clearance {result.min_clearance:+.3f} m (rect), "
                 f"{result.min_clearance_circle:+.3f} m (3 circles)")
    axes[1].plot(x, [math.degrees(r["delta_cmd"]) for r in log], label="delta cmd")
    axes[1].plot(x, [math.degrees(r["delta"]) for r in log], label="delta actual")
    axes[1].set_ylabel("wheel angle [deg]"); axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    axes[2].plot(x, [r["v"] for r in log], label="v")
    axes[2].plot(x, [r["e_y"] for r in log], label="e_y")
    axes[2].plot(x, [r["clr_rect"] for r in log], label="clearance (rect)")
    axes[2].set_ylim(-0.5, 2.0); axes[2].set_xlabel("x [m]"); axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=default_config_path())
    ap.add_argument("--seeds", type=int, default=0, help="Monte Carlo run count (0 = single run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gap", type=float, help="longitudinal pair spacing override [m]")
    ap.add_argument("--set", action="append", default=[], metavar="section.key=value")
    ap.add_argument("--plot", metavar="PNG")
    ap.add_argument("--csv", metavar="CSV")
    ap.add_argument("--workers", type=int)
    args = ap.parse_args(argv)

    cfg = build_config(args.config, args.set, args.gap)
    if args.seeds > 0:
        print(summarize(monte_carlo(cfg, args.seeds, args.workers)))
        return
    result = Simulator(cfg, args.seed, keep_log=True).run()
    print(summarize([result]))
    if args.csv:
        with open(args.csv, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result.log[0].keys()))
            writer.writeheader()
            writer.writerows(result.log)
    if args.plot:
        plot(result, cfg, args.plot)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# ROS synthetic vehicle (merged from sim_vehicle_node.py)
# ---------------------------------------------------------------------------

"""Hardware-free stand-in for the vehicle: bicycle plant + synthetic LiDAR / YOLO.

Replaces the external vehicle/CAN interface and sensors so the full ROS graph (perception -> mission ->
control) can be exercised on a desk:

    ros2 launch cone_avoidance sim.launch.py
"""


import math
from collections import deque

import numpy as np
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, Float32, Header, Int32
from tf2_ros import TransformBroadcaster

from .perception_node import synth_scan
from .params import DEFAULTS
from .ros_common import run_node, declare_config

LIDAR_X, LIDAR_Z = 0.6, 0.8          # sensor pose in base_link (rear axle)


class SimVehicleNode(Node):
    def __init__(self):
        super().__init__("cone_sim_vehicle")
        self.cfg = declare_config(self)
        self.declare_parameter("sim.cones", DEFAULTS["sim"]["cones"])
        self.cones = np.asarray(self.get_parameter("sim.cones").value, dtype=float).reshape(-1, 2)
        t = self.cfg.topics
        self.x = self.y = self.yaw = self.v = self.delta = 0.0
        self.steer_cmd = self.speed_cmd = 0.0
        self.dt = 0.02
        self.queue = deque([0.0] * 3)
        self.step_count = 0

        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(
            Float32, t.steer_cmd,
            lambda m: setattr(self, "steer_cmd", from_cmd_units(float(m.data), 0.0, self.cfg)[0]), 10)
        self.create_subscription(
            Float32, t.speed_cmd,
            lambda m: setattr(self, "speed_cmd", from_cmd_units(0.0, float(m.data), self.cfg)[1]), 10)
        self.odom_pub = self.create_publisher(Odometry, t.odometry, qos_profile_sensor_data)
        self.cloud_pub = self.create_publisher(PointCloud2, t.pointcloud, qos_profile_sensor_data)
        self.yolo_pub = self.create_publisher(Int32, t.yolo_count, 10)
        self.fb_pub = self.create_publisher(Float32, t.steer_feedback, 10)
        self.ws_pub = self.create_publisher(Float32, t.wheel_speed, 10)
        self.zero_pub = self.create_publisher(Bool, t.zero_done, latched)
        self.zero_pub.publish(Bool(data=True))
        self.tf_pub = TransformBroadcaster(self)   # RViz needs odom -> base_link for the cloud
        self.create_timer(self.dt, self.tick)

    def tick(self):
        cfg = self.cfg
        wb = cfg.vehicle.wheelbase
        self.queue.append(self.steer_cmd)
        target = self.queue.popleft()
        k = cfg.max_steer_rate / 0.125
        rate = max(-cfg.max_steer_rate, min(cfg.max_steer_rate, k * (target - self.delta)))
        self.delta += rate * self.dt
        self.v += max(-0.8 * self.dt, min(0.8 * self.dt, self.speed_cmd - self.v))
        self.x += self.v * math.cos(self.yaw) * self.dt
        self.y += self.v * math.sin(self.yaw) * self.dt
        yaw_rate = self.v / wb * math.tan(self.delta)
        self.yaw += yaw_rate * self.dt
        self.step_count += 1

        stamp = self.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id, odom.child_frame_id = stamp, "odom", "base_link"
        odom.pose.pose.position.x, odom.pose.pose.position.y = self.x, self.y
        odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = math.sin(self.yaw / 2), math.cos(self.yaw / 2)
        odom.twist.twist.linear.x, odom.twist.twist.angular.z = self.v, yaw_rate
        self.odom_pub.publish(odom)
        tf = TransformStamped()
        tf.header = odom.header
        tf.child_frame_id = "base_link"
        tf.transform.translation.x, tf.transform.translation.y = self.x, self.y
        tf.transform.rotation = odom.pose.pose.orientation
        self.tf_pub.sendTransform(tf)
        self.fb_pub.publish(Float32(data=float(self.delta)))
        self.ws_pub.publish(Float32(data=float(self.v)))
        if self.step_count % 5 == 0:
            self.cloud_pub.publish(self._cloud(stamp))
        if self.step_count % 6 == 0:
            self.yolo_pub.publish(Int32(data=self._yolo()))

    def _local_cones(self):
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        dx, dy = self.cones[:, 0] - self.x, self.cones[:, 1] - self.y
        return np.column_stack((c * dx + s * dy, -s * dx + c * dy))

    def _cloud(self, stamp) -> PointCloud2:
        local = self._local_cones() - np.array([LIDAR_X, 0.0])
        local = local[local[:, 0] > 0.3]
        # synth_scan works about the ground plane; shift into base_link afterwards.
        ground_z = self.cfg.frames.ground_z
        scan = synth_scan(local, self.cfg.cone_radius, LIDAR_Z - ground_z,
                          self.cfg.perception.cone_height)
        scan[:, 0] += LIDAR_X
        scan[:, 2] += ground_z
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                  for i, n in enumerate("xyz")]
        return point_cloud2.create_cloud(Header(stamp=stamp, frame_id="base_link"), fields,
                                         scan.tolist())

    def _yolo(self) -> int:
        local = self._local_cones()
        ahead = local[:, 0] - self.cfg.vehicle.wheelbase
        ang = np.degrees(np.arctan2(local[:, 1], np.maximum(ahead, 1e-3)))
        return int(np.sum((ahead > 0.3) & (ahead < 12.0) & (np.abs(ang) < 35.0)))


def sim_vehicle_main(args=None):
    run_node(SimVehicleNode)
