from glob import glob
import os

from setuptools import find_packages, setup


package_name = "stop_mission"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=("test",)),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="klu5109",
    maintainer_email="klu5109@todo.todo",
    description="KROWD Ouster cone perception and center-line planner",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "cone_perception = stop_mission.perception_node:main",
            "cone_planner = stop_mission.planner_node:main",
            "pure_pursuit_controller = stop_mission.controller_node:main",
            "pointcloud_hold = stop_mission.pointcloud_hold_node:main",
        ],
    },
)
