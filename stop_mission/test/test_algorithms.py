import math

import numpy as np

from stop_mission.algorithms import (
    cone_centroids,
    delaunay_midpoint_graph,
    path_yaw_curvature,
    resample_path,
)


def test_cone_clustering_separates_compact_objects():
    first = np.asarray([
        [2.00, 1.00, 0.10],
        [2.04, 1.01, 0.20],
        [1.98, 0.99, 0.35],
    ])
    second = np.asarray([
        [4.00, -1.00, 0.10],
        [4.03, -1.02, 0.25],
        [3.98, -0.98, 0.40],
    ])
    centers = cone_centroids(
        np.vstack((first, second)), 0.20, 3, 20, 0.50, 0.10
    )
    assert len(centers) == 2
    assert np.allclose(centers[0], (2.0066667, 1.0), atol=1.0e-5)


def test_delaunay_graph_uses_opposite_side_midpoints():
    points = np.asarray([[2.0, 2.0], [2.0, -2.0], [5.0, 2.0], [5.0, -2.0]])
    colors = ["blue", "yellow", "blue", "yellow"]
    graph, info, triangles = delaunay_midpoint_graph(points, colors, 5.0, 170.0)
    assert triangles
    assert graph
    assert all({value[2], value[3]} == {"blue", "yellow"} for value in info.values())


def test_path_resampling_and_curvature_are_finite():
    path = resample_path([(0.0, 0.0), (2.0, 0.0), (3.0, 1.0)], 0.2)
    yaw, curvature = path_yaw_curvature(path)
    assert len(path) > 10
    assert np.all(np.isfinite(yaw))
    assert np.all(np.isfinite(curvature))
    assert math.isclose(path[-1][0], 3.0)


def test_cone_clustering_rejects_objects_taller_than_a_cone():
    """Reeds and guardrail posts measure z > 0.46 in base_link; cones stay below."""
    cone = np.asarray([
        [3.00, 1.90, 0.10],
        [3.04, 1.91, 0.25],
        [2.98, 1.89, 0.40],
    ])
    reed = np.asarray([
        [3.00, -1.90, 0.20],
        [3.04, -1.91, 0.45],
        [2.98, -1.89, 0.70],
    ])
    both = np.vstack((cone, reed))
    assert len(cone_centroids(both, 0.20, 3, 20, 0.50, 0.10)) == 2
    kept = cone_centroids(both, 0.20, 3, 20, 0.50, 0.10, 0.50)
    assert len(kept) == 1
    assert np.allclose(kept[0], (3.0066667, 1.9), atol=1.0e-5)
