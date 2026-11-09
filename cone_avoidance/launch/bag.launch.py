"""Replay a recorded run through the mission stack, with RViz.

    ros2 launch cone_avoidance bag.launch.py bag:=/mnt/c/Users/tk205/Downloads/0919_avoidance_f

Nothing is sent to the vehicle: the CAN bridge is off and the control node's commands
stay on their topics. Add `paused:=true` to start paused (press SPACE in the terminal
to step through), `rate:=0.5` to slow the replay down, or `play:=false` to run
`ros2 bag play` yourself in another terminal.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression


def generate_launch_description():
    share = get_package_share_directory("cone_avoidance")
    bag = LaunchConfiguration("bag")
    paused = LaunchConfiguration("paused")

    return LaunchDescription([
        DeclareLaunchArgument("bag", description="rosbag2 directory to replay"),
        DeclareLaunchArgument("rate", default_value="1.0"),
        DeclareLaunchArgument("paused", default_value="false"),
        DeclareLaunchArgument("play", default_value="true"),
        DeclareLaunchArgument("config", default_value=os.path.join(
            share, "config", "cone_avoidance.yaml")),
        DeclareLaunchArgument("overrides", default_value=os.path.join(
            share, "config", "bag_replay.yaml"),
            description="parameter file applied on top of `config` (topic names of the recording)"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(share, "rviz", "cone_avoidance.rviz")),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(share, "launch", "cone_avoidance.launch.py")),
            launch_arguments={
                "config": LaunchConfiguration("config"),
                "overrides": LaunchConfiguration("overrides"),
                "use_sim_time": "true",     # follow the bag clock
                "use_can": "false",         # never drive the vehicle from a replay
                "use_rviz": LaunchConfiguration("use_rviz"),
                "rviz_config": LaunchConfiguration("rviz_config"),
            }.items(),
        ),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", bag, "--clock", "--rate", LaunchConfiguration("rate")],
            output="screen",
            condition=IfCondition(PythonExpression(
                ["'", LaunchConfiguration("play"), "' == 'true' and '", paused, "' != 'true'"])),
        ),
        ExecuteProcess(
            cmd=["ros2", "bag", "play", bag, "--clock", "--rate", LaunchConfiguration("rate"),
                 "--start-paused"],
            output="screen",
            condition=IfCondition(PythonExpression(
                ["'", LaunchConfiguration("play"), "' == 'true' and '", paused, "' == 'true'"])),
        ),
    ])
