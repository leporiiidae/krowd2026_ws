"""ROS-independent geometry used by stop_mission nodes."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree


Point2D = tuple[float, float]


def quaternion_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    norm = x * x + y * y + z * z + w * w
    if norm < 1.0e-12:
        return np.eye(3)
    scale = 2.0 / norm
    xx, yy, zz = x * x * scale, y * y * scale, z * z * scale
    xy, xz, yz = x * y * scale, x * z * scale, y * z * scale
    wx, wy, wz = w * x * scale, w * y * scale, w * z * scale
    return np.asarray([
        [1.0 - yy - zz, xy - wz, xz + wy],
        [xy + wz, 1.0 - xx - zz, yz - wx],
        [xz - wy, yz + wx, 1.0 - xx - yy],
    ])


def voxel_downsample(points: np.ndarray, size: float) -> np.ndarray:
    if len(points) == 0 or size <= 0.0:
        return points
    keys = np.floor(points / size).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def euclidean_clusters(points_xy: np.ndarray, tolerance: float) -> list[np.ndarray]:
    """Return connected components of an unordered XY point cloud."""
    if len(points_xy) == 0:
        return []
    if tolerance <= 0.0:
        raise ValueError("cluster tolerance must be positive")

    tree = cKDTree(points_xy)
    visited = np.zeros(len(points_xy), dtype=bool)
    clusters: list[np.ndarray] = []
    for seed in range(len(points_xy)):
        if visited[seed]:
            continue
        stack = [seed]
        visited[seed] = True
        component: list[int] = []
        while stack:
            index = stack.pop()
            component.append(index)
            for neighbor in tree.query_ball_point(points_xy[index], tolerance):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        clusters.append(np.asarray(component, dtype=np.int64))
    return clusters


def cone_centroids(
    points_xyz: np.ndarray,
    tolerance: float,
    min_points: int,
    max_points: int,
    max_diameter: float,
    min_height: float,
    max_top_z: float = float("inf"),
) -> list[Point2D]:
    """Cluster an ROI cloud and retain compact, vertically extended objects.

    `max_top_z` rejects anything whose highest point stands above a cone: reeds
    and guardrail posts. It only works while the ROI `max_z` crop stays above
    this value, otherwise the tall part is cropped away before it can be seen.
    """
    result: list[Point2D] = []
    for indices in euclidean_clusters(points_xyz[:, :2], tolerance):
        cluster = points_xyz[indices]
        if not min_points <= len(cluster) <= max_points:
            continue
        xy_span = np.ptp(cluster[:, :2], axis=0)
        if float(np.linalg.norm(xy_span)) > max_diameter:
            continue
        if float(np.ptp(cluster[:, 2])) < min_height:
            continue
        if float(np.max(cluster[:, 2])) > max_top_z:
            continue
        center = np.mean(cluster[:, :2], axis=0)
        result.append((float(center[0]), float(center[1])))
    return sorted(result, key=lambda point: (point[0], abs(point[1])))


def triangle_angles(edge_lengths: Sequence[float]) -> list[float]:
    if len(edge_lengths) != 3 or any(length <= 0.0 for length in edge_lengths):
        return [180.0]
    angles = []
    for index in range(3):
        a = edge_lengths[index]
        b = edge_lengths[(index + 1) % 3]
        c = edge_lengths[(index + 2) % 3]
        cosine = np.clip((a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)
        angles.append(float(math.degrees(math.acos(cosine))))
    return angles


def delaunay_midpoint_graph(
    points: np.ndarray,
    colors: Sequence[str],
    min_inner_angle: float,
    max_inner_angle: float,
) -> tuple[dict[Point2D, dict[Point2D, float]], dict[Point2D, tuple[np.ndarray, np.ndarray, str, str]], list[np.ndarray]]:
    graph: dict[Point2D, dict[Point2D, float]] = {}
    midpoint_info: dict[Point2D, tuple[np.ndarray, np.ndarray, str, str]] = {}
    try:
        triangulation = Delaunay(points)
    except (QhullError, ValueError):
        return graph, midpoint_info, []

    accepted: list[np.ndarray] = []
    for simplex in triangulation.simplices:
        if len({colors[int(index)] for index in simplex}) == 1:
            continue
        lengths = [
            float(np.linalg.norm(points[int(simplex[i])] - points[int(simplex[j])]))
            for i in range(3)
            for j in range(i + 1, 3)
        ]
        if any(
            angle < min_inner_angle or angle > max_inner_angle
            for angle in triangle_angles(lengths)
        ):
            continue
        accepted.append(simplex)
        mids: list[Point2D] = []
        for i in range(3):
            for j in range(i + 1, 3):
                first, second = int(simplex[i]), int(simplex[j])
                if {colors[first], colors[second]} != {"blue", "yellow"}:
                    continue
                midpoint_array = (points[first] + points[second]) * 0.5
                midpoint = (float(midpoint_array[0]), float(midpoint_array[1]))
                graph.setdefault(midpoint, {})
                midpoint_info[midpoint] = (
                    points[first], points[second], colors[first], colors[second]
                )
                mids.append(midpoint)
        for i, first in enumerate(mids):
            for second in mids[i + 1:]:
                distance = math.dist(first, second)
                graph.setdefault(first, {})[second] = distance
                graph.setdefault(second, {})[first] = distance
    return graph, midpoint_info, accepted


def resample_path(points: Sequence[Point2D], spacing: float) -> list[Point2D]:
    if len(points) < 2:
        return list(points)
    source = np.asarray(points, dtype=float)
    segment_lengths = np.linalg.norm(np.diff(source, axis=0), axis=1)
    arc = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    if arc[-1] <= 1.0e-6:
        return [points[0]]
    sample_arc = np.arange(0.0, arc[-1], max(spacing, 0.05))
    if len(sample_arc) == 0 or sample_arc[-1] < arc[-1]:
        sample_arc = np.append(sample_arc, arc[-1])
    x = np.interp(sample_arc, arc, source[:, 0])
    y = np.interp(sample_arc, arc, source[:, 1])
    return [(float(px), float(py)) for px, py in zip(x, y)]


def path_yaw_curvature(points: Sequence[Point2D]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(points, dtype=float)
    if len(array) < 2:
        return np.zeros(len(array)), np.zeros(len(array))
    dx = np.gradient(array[:, 0])
    dy = np.gradient(array[:, 1])
    yaw = np.unwrap(np.arctan2(dy, dx))
    arc_gradient = np.maximum(np.hypot(dx, dy), 1.0e-6)
    curvature = np.gradient(yaw) / arc_gradient
    return yaw, curvature


def nearest(nodes: Iterable[Point2D], point: Point2D) -> Point2D | None:
    candidates = list(nodes)
    return min(candidates, key=lambda candidate: math.dist(candidate, point)) if candidates else None
