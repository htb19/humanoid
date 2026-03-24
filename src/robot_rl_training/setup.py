from glob import glob
import os

from setuptools import find_packages, setup


package_name = "robot_rl_training"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="arthur",
    maintainer_email="arthur@example.com",
    description="Isaac Sim PPO training package for humanoid brick grasping.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "train_ppo = robot_rl_training.train_ppo:main",
            "eval_policy = robot_rl_training.eval_policy:main",
        ],
    },
)
