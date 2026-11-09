"""Diagnose a run from its control log: does the steering follow, does the car follow?

    python3 -m cone_avoidance.check_log logs/control_20260923_140000.csv

Separates the three failures that look identical from the driver's seat:
  * the wheel does not reach the commanded angle  -> steering / ratio / firmware
  * the wheel follows but the car leaves the path -> control gains, latency, speed
  * the car is on its path and still hits a cone  -> cone estimate / lane anchoring
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from typing import List, Optional

import numpy as np


def _col(rows: List[dict], name: str) -> np.ndarray:
    out = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        try:
            out[i] = float(r.get(name, ""))
        except (TypeError, ValueError):
            pass
    return out


def _lag_seconds(cmd: np.ndarray, fb: np.ndarray, dt: float, max_lag: float = 1.5) -> Optional[float]:
    """Shift that best lines the feedback up with the command (cross-correlation)."""
    good = np.isfinite(cmd) & np.isfinite(fb)
    if good.sum() < 50:
        return None
    c = cmd[good] - np.mean(cmd[good])
    f = fb[good] - np.mean(fb[good])
    if np.std(c) < 1e-6:
        return None
    best, best_score = 0, -np.inf
    for shift in range(0, int(max_lag / dt) + 1):
        a, b = c[: len(c) - shift], f[shift:]
        score = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        if score > best_score:
            best, best_score = shift, score
    return best * dt


def analyse(path: str) -> int:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print(f"{path}: empty")
        return 1

    t = _col(rows, "t")
    dt = float(np.median(np.diff(t))) if len(t) > 2 else 0.02
    cmd = _col(rows, "delta_wheel")
    fb = _col(rows, "steer_fb_wheel")
    e_y = _col(rows, "e_y")
    theta_e = _col(rows, "theta_e")
    v = _col(rows, "v")
    v_cmd = _col(rows, "v_cmd")
    states = [r.get("state", "") for r in rows]
    moving = np.isfinite(v) & (v > 0.2)

    print(f"file      : {path}")
    print(f"duration  : {t[-1] - t[0]:.1f} s, {len(rows)} rows, control period {dt * 1000:.1f} ms")
    print(f"states    : {' > '.join(dict.fromkeys(s for s in states if s))}")
    print(f"speed     : cmd {np.nanmax(v_cmd) * 3.6:.1f} km/h max, measured {np.nanmax(v) * 3.6:.1f} km/h max")
    print()

    # --- 1. does the wheel do what it is told? -------------------------------
    print("1) steering: command -> wheel")
    if not np.isfinite(fb).any():
        print("   NO FEEDBACK on /cone_avoidance/steer_feedback (0x201).")
        print("   -> cannot tell whether the wheel followed. Wire the feedback before trusting anything else.")
    else:
        big = np.isfinite(cmd) & np.isfinite(fb) & (np.abs(cmd) > math.radians(2.0))
        gain = float(np.sum(cmd[big] * fb[big]) / np.sum(cmd[big] ** 2)) if big.sum() > 20 else float("nan")
        lag = _lag_seconds(cmd, fb, dt)
        rate = np.abs(np.gradient(fb[np.isfinite(fb)], dt))
        rate_p95 = float(np.percentile(rate, 95)) if len(rate) else float("nan")
        rate_max = float(np.percentile(rate, 99.9)) if len(rate) else float("nan")
        err = np.abs(cmd - fb)
        print(f"   gain (wheel / command) : {gain:.2f}   (1.00 = exactly as commanded)")
        print(f"   lag                    : {lag:.2f} s" if lag is not None else "   lag: n/a")
        print(f"   rate achieved          : {math.degrees(rate_p95):.1f} deg/s (95pct), "
              f"{math.degrees(rate_max):.1f} deg/s (peak)")
        print(f"   |command - wheel|      : mean {math.degrees(np.nanmean(err)):.1f} deg, "
              f"max {math.degrees(np.nanmax(err)):.1f} deg")
        if np.isfinite(gain) and gain < 0.8:
            print(f"   => THE WHEEL TURNS {100 * (1 - gain):.0f}% LESS THAN COMMANDED.")
            print(f"      steering.output_ratio is probably {6.8 / max(gain, 0.05):.1f}, not 6.8,")
            print("      or the firmware clips (output_limit_rad) / loses steps.")
        if lag is not None and lag > 0.25:
            print(f"   => the wheel is {lag:.2f} s behind. Set control.latency_compensation to about {lag:.2f}")
            print("      and check the steering firmware (small errors moving at minimum step rate).")
        # A rate seen during a run is a DEMAND, not a capability: it is only the
        # actuator's limit when the wheel was lagging behind the command.
        if np.isfinite(rate_p95):
            lagging = float(np.nanmean(err)) > math.radians(2.0)
            if lagging:
                print(f"   => the wheel was saturated while lagging: steering.max_rate_deg_s is about "
                      f"{math.degrees(rate_max):.0f}, not {60}")
            else:
                print("   => the wheel kept up, so this run does not show its limit. "
                      "Measure it: ros2 run cone_avoidance steer_probe")

    # --- 2. does the car follow the path? ------------------------------------
    print("\n2) path following")
    if not np.isfinite(e_y).any():
        print("   no e_y in this log (the controller never had a path).")
    else:
        for label, mask in (("all", np.isfinite(e_y)),
                            ("moving", moving & np.isfinite(e_y)),
                            ("avoiding", np.array([s.startswith("AVOID") or s == "TRANSIT" for s in states])
                             & np.isfinite(e_y))):
            if mask.sum() < 5:
                continue
            print(f"   |e_y| {label:9s}: mean {np.nanmean(np.abs(e_y[mask])):.3f} m, "
                  f"max {np.nanmax(np.abs(e_y[mask])):.3f} m   "
                  f"|heading err| max {math.degrees(np.nanmax(np.abs(theta_e[mask]))):.1f} deg")
        worst = float(np.nanmax(np.abs(e_y[moving]))) if moving.any() else float("nan")
        if np.isfinite(worst) and worst > 0.25:
            print(f"   => the car is up to {worst:.2f} m off its own path. The plan was fine; the car was not on it.")
            print("      Look at (1) first: a slow or weak steering explains this. If (1) is clean,")
            print("      raise control.k_stanley or lower speed.target.")
        elif np.isfinite(worst):
            print(f"   => the car stayed on its path (max {worst:.2f} m). If it still hit a cone,")
            print("      the PATH was wrong: check the cone positions and lane anchoring in plan_*.csv.")

    # --- 3. what the planner thought ----------------------------------------
    clr = _col(rows, "clearance_rect")
    if np.isfinite(clr).any():
        print(f"\n3) planned clearance to the cones: min {np.nanmin(clr):.3f} m "
              f"(negative = the plan itself went through a cone)")
    flags = {f for r in rows for f in (r.get("flags") or "").split("|") if f}
    faults = {f for r in rows for f in (r.get("faults") or "").split("|") if f}
    if flags or faults:
        print(f"   flags: {', '.join(sorted(flags)) or '-'}   faults: {', '.join(sorted(faults)) or '-'}")
    return 0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="+", help="logs/control_*.csv")
    args = ap.parse_args(argv)
    for i, path in enumerate(args.csv):
        if i:
            print("\n" + "=" * 70 + "\n")
        analyse(path)


if __name__ == "__main__":
    sys.exit(main())
