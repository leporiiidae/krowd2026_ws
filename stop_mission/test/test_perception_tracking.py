import math

import numpy as np

from stop_mission.perception_node import ConePerceptionNode


class _Logger:
    def warn(self, *_args, **_kwargs):
        pass


def test_chain_motion_compensation_applies_relative_ekf_pose():
    translated = ConePerceptionNode.compensate_chain_point(
        (10.0, 2.0, 4), 1.0, 0.0, 0.0
    )
    assert translated == (9.0, 2.0, 4)

    rotated = ConePerceptionNode.compensate_chain_point(
        (10.0, 2.0, 4), 0.0, 0.0, math.radians(10.0)
    )
    assert rotated[1] < 2.0
    assert rotated[2] == 4


def test_chain_pruning_has_no_age_expiry():
    chain = [(-1.0, 0.0, 999), (1.0, 0.0, 999), (2.0, 0.0, 0)]
    retained = ConePerceptionNode.pruned_chain(chain, -0.5, 40)
    assert np.allclose([point[:2] for point in retained], [(1.0, 0.0), (2.0, 0.0)])
    assert retained[0][2] == 999


def test_sector_tracking_labels_current_detection_not_chain_coordinate():
    node = ConePerceptionNode.__new__(ConePerceptionNode)
    parameters = {
        "tracking_match_distance": 0.8,
        "sector_tail_size": 2,
        "side_sector_radius": 5.0,
        "side_sector_angle_deg": 35.0,
    }
    node.parameter = parameters.__getitem__
    node.left_chain = [(2.0, 2.0, 3), (4.0, 2.0, 1)]
    node.right_chain = [(2.0, -2.0, 3), (4.0, -2.0, 1)]
    current_detections = [(4.25, 2.05), (4.20, -2.10), (8.0, 7.0)]

    labels = node.classify_sector_only(current_detections)

    assert labels == ["blue", "yellow", "unknown"]
    # Classification returns labels only. The caller publishes the untouched
    # current_detections, never the historical chain coordinates.
    assert current_detections[0] == (4.25, 2.05)


def test_reinitialization_requires_current_cones_on_both_sides():
    node = ConePerceptionNode.__new__(ConePerceptionNode)
    parameters = {"initial_chain_min_length": 2, "seed_min_abs_y": 1.0}
    node.parameter = parameters.__getitem__

    assert node.can_reinitialize([(2.0, 2.0), (4.0, 2.2), (2.0, -2.0)]) is False
    assert node.can_reinitialize(
        [(2.0, 2.0), (4.0, 2.2), (2.0, -2.0), (4.0, -2.2)]
    ) is True


def test_large_ekf_yaw_jump_requests_lidar_reclassification():
    node = ConePerceptionNode.__new__(ConePerceptionNode)
    parameters = {
        "motion_min_dt": 0.005,
        "yaw_jump_threshold_deg": 5.0,
        "yaw_stable_threshold_deg": 1.0,
        "yaw_stable_frames": 5,
    }
    node.parameter = parameters.__getitem__
    node.interpolate_odometry_pose = lambda _stamp: (0.0, 0.0, math.radians(12.0))
    node.get_logger = lambda: _Logger()
    node.previous_motion_time = 0.0
    node.previous_motion_pose = (0.0, 0.0, 0.0)
    node.left_chain = [(5.0, 2.0, 0)]
    node.right_chain = [(5.0, -2.0, 0)]
    node.virtual_left = []
    node.virtual_right = []
    node.initialized = True
    node.tracking_loss_count = 0
    node.yaw_realigning = False
    node.yaw_stable_count = 0

    reclassify = node.compensate_chains_to_time(0.1)

    assert reclassify is True
    assert node.yaw_realigning is True
    assert node.initialized is False
    assert node.left_chain == []
    assert node.right_chain == []
    assert "yaw_jump=12.00deg" in node.last_motion_debug


def test_yaw_realignment_exits_only_after_required_stable_frames():
    node = ConePerceptionNode.__new__(ConePerceptionNode)
    parameters = {
        "motion_min_dt": 0.005,
        "yaw_jump_threshold_deg": 5.0,
        "yaw_stable_threshold_deg": 1.0,
        "yaw_stable_frames": 2,
    }
    node.parameter = parameters.__getitem__
    node.get_logger = lambda: _Logger()
    node.previous_motion_time = 0.0
    node.previous_motion_pose = (0.0, 0.0, 0.0)
    node.left_chain = []
    node.right_chain = []
    node.virtual_left = []
    node.virtual_right = []
    node.initialized = True
    node.tracking_loss_count = 0
    node.yaw_realigning = True
    node.yaw_stable_count = 0
    poses = iter([(0.0, 0.0, math.radians(0.5)), (0.0, 0.0, math.radians(0.8))])
    node.interpolate_odometry_pose = lambda _stamp: next(poses)

    assert node.compensate_chains_to_time(0.1) is True
    assert node.yaw_realigning is True
    assert node.compensate_chains_to_time(0.2) is True
    assert node.yaw_realigning is False
    assert "lidar_reclassify_final" in node.last_motion_debug
