"""Full ROS graph against the synthetic vehicle (no hardware, no CAN), with RViz."""

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
    use_rviz = LaunchConfiguration("use_rviz")
    sim = {"mission.auto_start": True, "frames.use_tf": False, "logging.path": "./logs_sim",
           # the synthetic world is laid out along +x from the origin
           "reference.mode": "fixed", "reference.origin_x": 1.34,
           "reference.origin_y": 0.0, "reference.heading_deg": 0.0}

    def node(executable, name, **kwargs):
        return Node(package="cone_avoidance", executable=executable, name=name,
                    output="screen", parameters=[config, sim], **kwargs)

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=os.path.join(share, "config", "cone_avoidance.yaml")),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(share, "rviz", "cone_avoidance.rviz")),
        DeclareLaunchArgument("rviz_software_gl", default_value="true" if _on_wsl() else "false"),
        node("sim_vehicle", "cone_sim_vehicle"),
        node("perception", "cone_perception"),
        node("mission", "cone_mission"),
        node("control", "cone_control"),
        node("pointcloud_hold", "cone_pointcloud_hold", condition=IfCondition(use_rviz)),
        Node(package="rviz2", executable="rviz2", name="cone_avoidance_rviz", output="screen",
             arguments=["-d", LaunchConfiguration("rviz_config")],
             additional_env={"LIBGL_ALWAYS_SOFTWARE": PythonExpression(
                 ["'1' if '", LaunchConfiguration("rviz_software_gl"), "' == 'true' else '0'"])},
             condition=IfCondition(use_rviz)),
    ])
