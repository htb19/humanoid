from glob import glob
import os

from setuptools import find_packages, setup

package_name = "robot_imitation_pipeline"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="arthur",
    maintainer_email="3098016342@qq.com",
    description="Real-robot imitation learning data pipeline tools.",
    license="TODO",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "demo_recorder_node = robot_imitation_pipeline.nodes.demo_recorder_node:main",
            "demo_control = robot_imitation_pipeline.demo_control:main",
            "validate_demo = robot_imitation_pipeline.validate_demo:main",
            "replay_demo = robot_imitation_pipeline.replay_demo:main",
            "convert_to_hdf5 = robot_imitation_pipeline.convert_to_hdf5:main",
            "train_bc = robot_imitation_pipeline.train_bc:main",
        ],
    },
)
