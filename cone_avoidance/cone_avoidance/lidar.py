"""LiDAR cone detection on a base_link point cloud (numpy / scipy only)."""

from __future__ import annotations

import math
from typing import List

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def min_points(dist: float, min_pts_at_5m: float) -> int:
    """Point-count threshold falls with range squared (16-ch ring spacing)."""
    return max(2, int(round(min_pts_at_5m * (5.0 / max(dist, 0.5)) ** 2)))


def deskew(xyz: np.ndarray, t_rel: np.ndarray, v: float, yaw_rate: float) -> np.ndarray:
    """Express every point in the base_link pose at t_rel = 0.

    A point measured ``t`` seconds later was seen from a pose advanced by the
    vehicle motion, so it is mapped back through that motion (planar, constant
    twist over one scan).
    """
    theta = yaw_rate * t_rel
    c, s = np.cos(theta), np.sin(theta)
    # Pose advance over t (arc; reduces to v*t straight ahead when yaw_rate -> 0).
    px = v * t_rel * np.sinc(theta / np.pi)
    py = v * t_rel * 0.5 * theta * np.sinc(theta / (2 * np.pi)) ** 2
    out = xyz.copy()
    out[:, 0] = c * xyz[:, 0] - s * xyz[:, 1] + px
    out[:, 1] = s * xyz[:, 0] + c * xyz[:, 1] + py
    return out


def cluster_xy(xy: np.ndarray, eps: float) -> np.ndarray:
    """Single-link clustering: labels of connected components under distance eps."""
    n = len(xy)
    if n == 0:
        return np.zeros(0, dtype=int)
    pairs = cKDTree(xy).query_pairs(eps, output_type="ndarray")
    if len(pairs) == 0:
        return np.arange(n)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def fit_center(xy: np.ndarray, radius: float) -> np.ndarray:
    """Centre of a known-radius cylinder from points on its sensor-facing arc.

    The centroid of the visible arc sits ~0.785 r in front of the axis, so the
    start value is pushed along the ray; a few Gauss-Newton steps refine it.
    """
    centroid = xy.mean(axis=0)
    ray = centroid / max(np.linalg.norm(centroid), 1e-6)
    center = centroid + 0.785 * radius * ray
    if len(xy) < 3:
        return center
    for _ in range(4):
        diff = xy - center
        rng = np.maximum(np.linalg.norm(diff, axis=1), 1e-6)
        jac = -diff / rng[:, None]
        res = rng - radius
        step, *_ = np.linalg.lstsq(jac, -res, rcond=None)
        if not np.all(np.isfinite(step)) or np.linalg.norm(step) > radius:
            break  # ill-conditioned (nearly collinear points): keep the ray-biased start
        center = center + step
    return center


def _width(xy: np.ndarray) -> float:
    """Extent perpendicular to the sensor ray."""
    centroid = xy.mean(axis=0)
    ray = centroid / max(np.linalg.norm(centroid), 1e-6)
    lateral = xy @ np.array([-ray[1], ray[0]])
    return float(lateral.max() - lateral.min())


def _row_axis(xy: np.ndarray) -> np.ndarray:
    """Unit vector along a row of cones (principal axis of the cluster)."""
    centred = xy - xy.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[0]


def _split_row(xy: np.ndarray, cfg) -> List[np.ndarray]:
    """Cone centres of a cluster wider than one cone.

    Cones standing side by side (touching, or closer than cluster_eps) arrive as ONE
    cluster. The count comes from the extent along the row, the members from 1-D
    k-means along it. Afterwards the two end cones are pushed outward, if needed,
    until they cover the cluster's extreme points: the planner steers by the inner
    edge of the row, so the row's extent must never be under-estimated.
    """
    p = cfg.perception
    radius = p.cone_diameter / 2.0
    axis = _row_axis(xy)
    t = xy @ axis
    extent = float(t.max() - t.min())
    # Real rows are not perfectly tight (measured 1.52 m for three 0.45 m cones).
    k = max(2, int(round(extent / (p.cone_diameter * p.row_pitch_ratio))))
    if k > int(p.max_cones_per_cluster):
        return []  # wall / guardrail, not a row of cones
    centers = np.linspace(t.min(), t.max(), 2 * k + 1)[1::2]
    for _ in range(8):
        label = np.argmin(np.abs(t[:, None] - centers[None, :]), axis=1)
        centers = np.array([t[label == j].mean() if np.any(label == j) else centers[j]
                            for j in range(k)])
    order = np.argsort(centers)
    fitted = []
    for j in order:
        part = xy[label == j]
        if len(part) == 0:
            continue
        dist = float(np.linalg.norm(part.mean(axis=0)))
        if len(part) < min_points(dist, p.min_pts_at_5m):
            continue
        fitted.append(fit_center(part, radius))
    if not fitted:
        return []
    low = float(fitted[0] @ axis) - radius - float(t.min())
    if low > 0.0:
        fitted[0] = fitted[0] - low * axis
    high = float(t.max()) - (float(fitted[-1] @ axis) + radius)
    if high > 0.0:
        fitted[-1] = fitted[-1] + high * axis
    return fitted


def _looks_like_cone(center: np.ndarray, column: np.ndarray, cfg) -> bool:
    """Height test on ALL returns above the footprint (z relative to the ground).

    Roadside vegetation and posts pass the width test all the time; what separates
    them is height: a cone ends at cone_height, and (close enough for the rings to be
    dense) it does reach up to near cone_height.
    """
    p = cfg.perception
    if len(column) == 0:
        return True
    top = float(column.max())
    if top > p.cone_height + p.tall_reject_margin:
        return False                                   # something taller stands here
    dist = float(np.linalg.norm(center))
    if dist < p.top_check_range and top < p.cone_height - p.top_min_below:
        return False                                   # too short to be a cone
    return True


def detect_cones(xyz: np.ndarray, cfg, ground_z: float = 0.0) -> np.ndarray:
    """Cone axis positions (M, 2) in base_link from one (deskewed) scan."""
    p = cfg.perception
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2] - ground_z
    sx0, sx1, sy = p.self_box
    roi = ((x >= p.roi_x[0]) & (x <= p.roi_x[1]) & (y >= p.roi_y[0]) & (y <= p.roi_y[1])
           & np.isfinite(x) & ~((x >= sx0) & (x <= sx1) & (np.abs(y) <= sy)))   # own bodywork
    keep = roi & (z >= p.ground_z[0]) & (z <= p.ground_z[1])
    xy = xyz[keep, :2]
    if len(xy) == 0:
        return np.zeros((0, 2))
    # Everything above the ground band, for the height test (includes tall objects).
    above = roi & (z >= p.ground_z[0]) & (z <= p.cone_height + 2.0)
    col_xy, col_z = xyz[above, :2], z[above]
    col_tree = cKDTree(col_xy) if len(col_xy) else None

    diameter = p.cone_diameter
    w_min, w_max = p.width_ratio_min * diameter, p.width_ratio_max * diameter
    labels = cluster_xy(xy, p.cluster_eps)
    out = []
    for label in np.unique(labels):
        pts = xy[labels == label]
        axis = _row_axis(pts) if len(pts) >= 3 else np.array([0.0, 1.0])
        along = pts @ axis
        if max(_width(pts), float(along.max() - along.min())) > w_max:
            candidates = _split_row(pts, cfg)
        else:
            dist = float(np.linalg.norm(pts.mean(axis=0)))
            if len(pts) < min_points(dist, p.min_pts_at_5m):
                continue
            # Far cones are hit by very few columns, so only the upper bound is firm.
            if _width(pts) < w_min and dist < 8.0:
                continue
            candidates = [fit_center(pts, diameter / 2.0)]
        for center in candidates:
            idx = col_tree.query_ball_point(center, diameter * 0.75) if col_tree is not None else []
            if _looks_like_cone(center, col_z[idx], cfg):
                out.append(center)
    return np.array(out).reshape(-1, 2)


def synth_scan(cones_xy, radius: float, sensor_z: float = 0.8, height: float = 0.8,
               az_step_deg: float = 0.35, rings_deg=None, max_range: float = 25.0) -> np.ndarray:
    """Ray-cast cylinders from a sensor at the origin (testing / sim_vehicle only).

    Every ray keeps its NEAREST hit, so cones standing shoulder to shoulder occlude
    each other the way they do for a real spinning LiDAR.
    """
    cones = np.asarray(cones_xy, dtype=float).reshape(-1, 2)
    rings = np.radians(np.linspace(-15.0, 15.0, 16) if rings_deg is None else rings_deg)
    if len(cones) == 0:
        return np.zeros((0, 3))
    az = np.arange(-math.pi / 2, math.pi / 2, math.radians(az_step_deg))
    ux, uy = np.cos(az), np.sin(az)
    b = ux[:, None] * cones[None, :, 0] + uy[:, None] * cones[None, :, 1]
    c = (cones ** 2).sum(axis=1)[None, :] - radius * radius
    disc = b * b - c
    rng = np.where((disc >= 0) & (b > 0), b - np.sqrt(np.maximum(disc, 0.0)), np.inf)
    rng = np.where(rng > 0, rng, np.inf).min(axis=1)
    hit = np.isfinite(rng) & (rng <= max_range)
    pts = []
    for r, x, y in zip(rng[hit], ux[hit], uy[hit]):
        z = sensor_z + r * np.tan(rings)
        for zz in z[(z >= 0.0) & (z <= height)]:
            pts.append((r * x, r * y, float(zz)))
    return np.array(pts).reshape(-1, 3)
