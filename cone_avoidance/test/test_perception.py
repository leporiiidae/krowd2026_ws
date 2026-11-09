import math

import numpy as np
import pytest

from cone_avoidance.perception_node import deskew, detect_cones, min_points, synth_scan
from cone_avoidance.params import Config
from cone_avoidance.perception_node import ConeTracker, RowFilter, Track, find_rows, group_pairs

CFG = Config()


def cylinder_points(cx, cy, radius=0.225, az_step_deg=0.35, rings=np.radians(np.linspace(-15, 15, 16)),
                    lidar_z=0.8, height=0.8):
    pts = []
    dist = math.hypot(cx, cy)
    centre, half = math.atan2(cy, cx), math.asin(radius / dist)
    for az in np.arange(centre - half, centre + half, math.radians(az_step_deg)):
        ux, uy = math.cos(az), math.sin(az)
        b = ux * cx + uy * cy
        disc = b * b - (dist * dist - radius * radius)
        if disc < 0:
            continue
        rng = b - math.sqrt(disc)
        for z in lidar_z + rng * np.tan(rings):
            if 0.0 <= z <= height:
                pts.append((rng * ux, rng * uy, z))
    return np.array(pts)


def test_min_points_scales_with_inverse_square_range():
    assert min_points(5.0, 15) == 15
    assert min_points(10.0, 15) == 4
    assert min_points(15.0, 15) == 2
    assert min_points(40.0, 15) == 2


def _ground(n=400):
    rng = np.random.default_rng(0)
    return np.column_stack((rng.uniform(1, 18, n), rng.uniform(-3, 3, n), np.zeros(n)))


@pytest.mark.parametrize("dist", [4.0, 8.0, 12.0])
@pytest.mark.parametrize("row", [[1.675, 1.225, 0.775], [-0.775, -1.225, -1.675]])
def test_three_touching_cones_are_split(dist, row):
    """Touching cones arrive as ONE cluster and partly occlude each other."""
    truth = np.array([[dist, d] for d in row])
    found = detect_cones(np.vstack((synth_scan(truth, 0.225), _ground())), CFG, 0.0)
    assert len(found) == 3
    found = found[np.argsort(found[:, 1])]
    truth = truth[np.argsort(truth[:, 1])]
    assert np.abs(found - truth).max() < 0.12
    # The inner edge (the one the planner steers around) is never under-estimated.
    inner = np.argmin(np.abs(truth[:, 1]))
    assert abs(found[inner, 1]) <= abs(truth[inner, 1]) + 0.03


def test_detection_uses_the_configured_ground_height():
    """base_link sits 0.27 m above the ground, so the z band is measured from -0.27.

    Getting the sign wrong lifts the band off the cone: near cones still survive, but
    the far ones the mission needs (detect_range 15 m) are lost.
    """
    cfg = Config()                                        # ground_z = -0.27
    ground = np.column_stack((np.random.default_rng(0).uniform(1, 18, 600),
                              np.random.default_rng(1).uniform(-3, 3, 600), np.full(600, -0.27)))
    for dist in (6.0, 10.0, 13.0):
        cone = synth_scan([[dist, 0.5]], 0.225, 0.8 + 0.27, 0.8)
        cone[:, 2] -= 0.27                                # express in base_link
        cloud = np.vstack((cone, ground))
        found = detect_cones(cloud, cfg, cfg.frames.ground_z)
        assert len(found) == 1 and np.abs(found[0] - [dist, 0.5]).max() < 0.08
        if dist >= 10.0:
            assert len(detect_cones(cloud, cfg, +0.27)) == 0   # sign flipped


def _at_ground(points, ground=-0.27):
    points = np.array(points, dtype=float)
    points[:, 2] += ground
    return points


def test_tall_and_short_objects_are_not_cones():
    rng = np.random.default_rng(3)
    def blob(cx, cy, top, n=120):                          # cone-WIDTH clutter of a given height
        return np.column_stack((rng.normal(cx, 0.08, n), rng.normal(cy, 0.08, n), rng.uniform(0.12, top, n)))
    post = _at_ground(blob(6.0, 1.0, 1.8))                  # taller than a cone: post / reeds
    bush = _at_ground(blob(7.0, -1.0, 0.40))                # too short within top_check_range
    cone = _at_ground(synth_scan([[6.5, 0.0]], 0.225, 0.8 + 0.27, 0.8))
    found = detect_cones(np.vstack((post, bush, cone)), CFG, CFG.frames.ground_z)
    assert len(found) == 1 and np.abs(found[0] - [6.5, 0.0]).max() < 0.08


def test_own_bodywork_is_ignored():
    hood = _at_ground(np.column_stack((np.full(80, 2.2), np.linspace(-0.2, 0.2, 80), np.full(80, 0.6))))
    assert len(detect_cones(hood, CFG, CFG.frames.ground_z)) == 0


def test_single_cone_is_still_one_cone():
    found = detect_cones(synth_scan([[6.0, 0.5]], 0.225), CFG)
    assert len(found) == 1 and np.abs(found[0] - [6.0, 0.5]).max() < 0.06


def test_wall_is_not_a_cone():
    wall = np.column_stack((np.full(300, 8.0), np.linspace(-2.5, 2.5, 300), np.full(300, 0.5)))
    assert len(detect_cones(wall, CFG)) == 0


def test_deskew_maps_late_points_back_to_scan_start_frame():
    # Static point 10 m ahead. At t the vehicle has advanced v*t, so it measures 10 - v*t.
    v, t = 2.0, 0.1
    measured = np.array([[10.0 - v * t, 0.0, 0.5]])
    out = deskew(measured, np.array([t]), v, 0.0)
    assert out[0, :2] == pytest.approx([10.0, 0.0])
    turned = deskew(np.array([[5.0, 0.0, 0.5]]), np.array([0.1]), 0.0, 0.5)   # pure yaw 0.05 rad
    assert math.atan2(turned[0, 1], turned[0, 0]) == pytest.approx(0.05)


def test_track_confirms_after_consecutive_frames_and_filters_position():
    tr = ConeTracker(CFG)
    assert tr.update([(10.0, 1.0)]) == []
    tr.update([(10.0, 1.0)])
    confirmed = tr.update([(10.2, 1.0)])
    assert len(confirmed) == 1
    assert confirmed[0].x == pytest.approx(10.0 + CFG.perception.track_filter_alpha * 0.2)


def test_unconfirmed_clutter_expires_but_confirmed_cone_is_kept_when_lost():
    tr = ConeTracker(CFG)
    for _ in range(3):
        tr.update([(10.0, 1.0)])
    tr.update([(10.0, 1.0), (4.0, -2.0)])                # one-frame clutter
    for _ in range(CFG.perception.miss_tolerance + 5):
        tr.update([])                                    # everything unseen (cone beside / behind)
    assert [t.track_id for t in tr.confirmed()] == [0]
    assert len(tr.tracks) == 1


def test_pairs_are_grouped_longitudinally():
    sd = np.array([[15.2, 0.8], [10.0, -1.4], [10.3, -0.8], [15.0, 1.4]])
    pairs = group_pairs([7, 8, 9, 10], sd, 1.5)
    assert [sorted(p.track_ids) for p in pairs] == [[8, 9], [7, 10]]
    assert pairs[1].s_min == pytest.approx(15.0)


ROW = [(10.0, 1.675), (10.0, 1.225), (10.0, 0.775)]      # group 1 (left), heading 0 = along +x
ROW2 = [(15.0, -0.775), (15.0, -1.225), (15.0, -1.675)]  # group 2 (right, against the solid line)


def test_row_of_three_across_the_road_is_kept():
    assert [sorted(r) for r in find_rows(np.array(ROW), 0.0, CFG)] == [[0, 1, 2]]


def test_row_along_the_road_is_not_an_obstacle():
    along = [(10.0, 1.2), (10.45, 1.2), (10.9, 1.2)]     # a hedge line beside the road
    assert find_rows(np.array(along), 0.0, CFG) == []


def test_dense_clutter_patch_is_not_an_obstacle_row():
    """Vegetation that happens to line up: measured 4-5 neighbours around such rows."""
    patch = ROW + [(9.2, 1.2), (9.3, 0.6), (10.9, 1.0), (10.8, 0.4)]
    assert find_rows(np.array(patch), 0.0, CFG) == []


def test_one_object_near_a_row_never_hides_it():
    """Rejecting a row because something stands next to it would hide both obstacles."""
    in_front = ROW + [(9.2, 1.2)]                         # e.g. a fallen cone in the lane
    assert [sorted(r) for r in find_rows(np.array(in_front), 0.0, CFG)] == [[0, 1, 2]]
    inward = ROW + [(10.3, 0.0)]                          # fused on the lane side
    rows = find_rows(np.array(inward), 0.0, CFG)
    assert len(rows) == 1 and set(rows[0]) == {0, 1, 2, 3}   # kept, AND reported as obstacle


def test_roadside_growth_beyond_the_line_never_hides_a_group():
    """Group 2 stands against the solid line: grass on the shoulder must not delete it."""
    shoulder = ROW2 + [(15.0, -2.35), (15.3, -2.9), (14.6, -2.6)]
    rows = find_rows(np.array(shoulder), 0.0, CFG)
    assert [sorted(r) for r in rows] == [[0, 1, 2]]        # exactly the three cones, not the grass


def test_duplicate_track_is_the_same_cone_not_clutter():
    dup = ROW + [(10.1, 1.25)]
    assert len(find_rows(np.array(dup), 0.0, CFG)) == 1


def test_partial_row_counts_but_a_single_does_not():
    # A far group often shows only 2 of its 3 cones; missing it would shift the numbering.
    assert len(find_rows(np.array(ROW[:2]), 0.0, CFG)) == 1
    assert find_rows(np.array([(8.0, 0.0)]), 0.0, CFG) == []


def test_accepted_row_stays_accepted():
    """Clutter appearing next to an accepted row later must not make it disappear."""
    rf = RowFilter(CFG)
    tracks = [Track(i, x, y, confirmed=True) for i, (x, y) in enumerate(ROW)]
    assert len(rf.update(tracks, 0.0)) == 3
    later = tracks + [Track(9, 9.2, 1.2, confirmed=True), Track(10, 9.3, 0.6, confirmed=True),
                      Track(11, 10.9, 1.0, confirmed=True)]      # would reject a NEW row
    assert sorted(t.track_id for t in rf.update(later, math.radians(20))) == [0, 1, 2]


def test_row_far_outside_the_lane_is_not_an_obstacle():
    beside = [(2.0, -4.0), (2.0, -4.45), (2.0, -4.9)]     # seen in the recording: grass 4 m off
    assert find_rows(np.array(beside), 0.0, CFG) == []
    far = [(18.0, -1.3), (18.0, -1.75), (18.0, -2.2)]    # a real group far ahead is still in reach
    assert len(find_rows(np.array(far), math.radians(8), CFG)) == 1
