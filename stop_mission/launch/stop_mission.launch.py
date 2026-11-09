from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    default_config = os.path.join(
        get_package_share_directory("stop_mission"), "config", "stop_mission.yaml"
    )
    default_rviz_config = os.path.join(
        get_package_share_directory("stop_mission"), "rviz", "stop_mission.rviz"
    )
    config = LaunchConfiguration("config")
    use_sim_time = LaunchConfiguration("use_sim_time")
    use_rviz = LaunchConfiguration("use_rviz")
    rviz_config = LaunchConfiguration("rviz_config")
    common = {"use_sim_time": use_sim_time}
    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Set true when replaying a stop_mission rosbag with --clock.",
        ),
        DeclareLaunchArgument(
            "use_rviz",
            default_value="true",
            description="Launch the dedicated stop-mission RViz2 window.",
        ),
        DeclareLaunchArgument(
            "rviz_config",
            default_value=default_rviz_config,
            description="RViz2 config for stop_mission visualization.",
        ),
        Node(
            package="stop_mission",
            executable="cone_perception",
            name="stop_mission_cone_perception",
            output="screen",
            parameters=[config, common],
        ),
        Node(
            package="stop_mission",
            executable="pointcloud_hold",
            name="stop_mission_pointcloud_hold",
            output="screen",
            parameters=[config, common],
        ),
        Node(
            package="stop_mission",
            executable="cone_planner",
            name="stop_mission_cone_planner",
            output="screen",
            parameters=[config, common],
        ),
        Node(
            package="stop_mission",
            executable="pure_pursuit_controller",
            name="stop_mission_pure_pursuit",
            output="screen",
            parameters=[config, common],
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="stop_mission_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[common],
            condition=IfCondition(use_rviz),
        ),
    ])
