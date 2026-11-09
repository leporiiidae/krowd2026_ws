import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
    # Second parameter file, applied on top of `config` (defaults to the same file, which
    # is a harmless duplicate). Bag replay uses it to override a few topics.
    overrides = LaunchConfiguration("overrides")
    use_rviz = LaunchConfiguration("use_rviz")
    common = {"use_sim_time": LaunchConfiguration("use_sim_time")}

    def node(executable, name, **kwargs):
        return Node(package="cone_avoidance", executable=executable, name=name,
                    output="screen", parameters=[config, overrides, common], **kwargs)

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=os.path.join(
            share, "config", "cone_avoidance.yaml")),
        DeclareLaunchArgument("overrides", default_value=config),
        DeclareLaunchArgument("use_sim_time", default_value="false",
                              description="true when replaying a rosbag with --clock"),
        DeclareLaunchArgument("use_can", default_value="true",
                              description="false for bag replay or when another CAN bridge runs"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(
            share, "rviz", "cone_avoidance.rviz")),
        # WSL's GPU OpenGL path makes RViz segfault after ~10 s; software GL is stable.
        DeclareLaunchArgument("rviz_software_gl", default_value="true" if _on_wsl() else "false"),
        node("perception", "cone_perception"),
        node("mission", "cone_mission"),
        node("control", "cone_control"),
        node("pointcloud_hold", "cone_pointcloud_hold", condition=IfCondition(use_rviz)),
        node("can_bridge", "cone_can", condition=IfCondition(LaunchConfiguration("use_can"))),
        Node(package="rviz2", executable="rviz2", name="cone_avoidance_rviz", output="screen",
             arguments=["-d", LaunchConfiguration("rviz_config")], parameters=[common],
             additional_env={"LIBGL_ALWAYS_SOFTWARE": PythonExpression(
                 ["'1' if '", LaunchConfiguration("rviz_software_gl"), "' == 'true' else '0'"])},
             condition=IfCondition(use_rviz)),
    ])
