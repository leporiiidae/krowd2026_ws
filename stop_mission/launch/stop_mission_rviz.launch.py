from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import os


def generate_launch_description():
    default_rviz_config = os.path.join(
        get_package_share_directory("stop_mission"), "rviz", "stop_mission.rviz"
    )
    rviz_config = LaunchConfiguration("rviz_config")
    use_sim_time = LaunchConfiguration("use_sim_time")

    return LaunchDescription([
        DeclareLaunchArgument(
            "rviz_config",
            default_value=default_rviz_config,
            description="RViz2 config for stop_mission visualization.",
        ),
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use rosbag /clock; set false for live vehicle data.",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            name="stop_mission_rviz",
            output="screen",
            arguments=["-d", rviz_config],
            parameters=[{"use_sim_time": use_sim_time}],
        ),
    ])
