"""Self-contained rosbag replay launch for the cone-avoidance stack.

Usage::

    ros2 launch cone_avoidance bag.launch.py bag:=/path/to/rosbag

The replay-specific topic and logging overrides live here, so a separate
``bag_replay.yaml`` is not required. CAN is intentionally never launched.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def _on_wsl() -> bool:
    try:
        with open("/proc/version", encoding="utf-8") as handle:
            return "microsoft" in handle.read().lower()
    except OSError:
        return False


def generate_launch_description():
    share = get_package_share_directory("cone_avoidance")
    config = LaunchConfiguration("config")
    use_rviz = LaunchConfiguration("use_rviz")
    bag = LaunchConfiguration("bag")
    paused = LaunchConfiguration("paused")

    common = {"use_sim_time": True}
    replay = {
        "topics.navsat": "/ublox_gps_node/fix",
        "logging.path": "./logs_bag",
    }

    def node(executable, name, **kwargs):
        return Node(
            package="cone_avoidance",
            executable=executable,
            name=name,
            output="screen",
            parameters=[config, replay, common],
            **kwargs,
        )

    return LaunchDescription([
        DeclareLaunchArgument("bag", description="rosbag2 directory to replay"),
        DeclareLaunchArgument("rate", default_value="1.0"),
        DeclareLaunchArgument("paused", default_value="false"),
        DeclareLaunchArgument("play", default_value="true"),
        DeclareLaunchArgument(
            "config",
            default_value=os.path.join(share, "config", "cone_avoidance.yaml"),
        ),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=os.path.join(share, "rviz", "cone_avoidance.rviz"),
        ),
        DeclareLaunchArgument(
            "rviz_software_gl",
            default_value="true" if _on_wsl() else "false",
        ),
        node("perception", "cone_perception"),
        node("mission", "cone_mission"),
        node("control", "cone_control"),
        node(
            "pointcloud_hold",
            "cone_pointcloud_hold",
            condition=IfCondition(use_rviz),
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="cone_avoidance_rviz",
            output="screen",
            arguments=["-d", LaunchConfiguration("rviz_config")],
            parameters=[common],
            additional_env={
                "LIBGL_ALWAYS_SOFTWARE": PythonExpression([
                    "'1' if '",
                    LaunchConfiguration("rviz_software_gl"),
                    "' == 'true' else '0'",
                ])
            },
            condition=IfCondition(use_rviz),
        ),
        ExecuteProcess(
            cmd=[
                "ros2", "bag", "play", bag, "--clock", "--rate",
                LaunchConfiguration("rate"),
            ],
            output="screen",
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration("play"), "' == 'true' and '",
                paused, "' != 'true'",
            ])),
        ),
        ExecuteProcess(
            cmd=[
                "ros2", "bag", "play", bag, "--clock", "--rate",
                LaunchConfiguration("rate"), "--start-paused",
            ],
            output="screen",
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration("play"), "' == 'true' and '",
                paused, "' == 'true'",
            ])),
        ),
    ])
