from glob import glob
import os

from setuptools import find_packages, setup

package_name = "cone_avoidance"

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
    maintainer="krowd2026",
    maintainer_email="krowd2026@gmail.com",
    description="Static cone-pair avoidance (zigzag) for the KROWD vehicle",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "perception = cone_avoidance.perception_node:main",
            "mission = cone_avoidance.mission_node:main",
            "control = cone_avoidance.control_node:main",
            "can_bridge = cone_avoidance.can_node:main",
            "sim_vehicle = cone_avoidance.sim:sim_vehicle_main",
            "pointcloud_hold = cone_avoidance.pointcloud_hold_node:main",
            "sim = cone_avoidance.sim:main",
            "check_log = cone_avoidance.check_log:main",
            "steer_probe = cone_avoidance.steer_probe_node:main",
        ],
    },
)
