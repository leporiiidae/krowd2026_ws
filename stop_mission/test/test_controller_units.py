import math

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

from stop_mission.controller_node import PurePursuitControllerNode


def test_target_speed_mps_for_can_interface():
    node = PurePursuitControllerNode.__new__(PurePursuitControllerNode)
    node.parameter = {"target_speed_mps": 4.0}.__getitem__

    assert math.isclose(node.target_speed_mps(), 4.0)


def test_braking_confirmation_requires_consecutive_frames():
    count = 0
    for distance in (2.9, 2.8, 3.1, 3.0, 2.7):
        count = PurePursuitControllerNode.next_braking_confirmation_count(
            distance, 3.0, count
        )

    assert count == 2


def test_braking_confirmation_reaches_ten_at_or_below_three_metres():
    count = 0
    for _ in range(10):
        count = PurePursuitControllerNode.next_braking_confirmation_count(
            3.0, 3.0, count
        )

    assert count == 10


def test_lookahead_selects_first_path_point_at_requested_distance():
    node = PurePursuitControllerNode.__new__(PurePursuitControllerNode)
    node.parameter = {"lookahead_distance": 3.0}.__getitem__
    node.path = Path()
    for x, y in ((1.0, 0.0), (2.8, 0.0), (3.1, 0.2), (4.0, 0.0)):
        pose = PoseStamped()
        pose.pose.position.x = x
        pose.pose.position.y = y
        node.path.poses.append(pose)

    assert node.select_target() == (3.1, 0.2)
