import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    robot_rl_share = Path(get_package_share_directory("robot_rl_training"))
    workspace_root = robot_rl_share.parents[3]
    trainer_script = workspace_root / "src" / "robot_rl_training" / "robot_rl_training" / "train_ppo.py"
    isaac_python = Path.home() / "isaacsim" / "python.sh"
    moveit_launch = Path(get_package_share_directory("robot_moveit_config")) / "launch" / "move_group.launch.py"
    source_root = workspace_root / "src" / "robot_rl_training"
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_entries = [str(source_root), str(workspace_root / "src")]
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)

    isaac_headless = DeclareLaunchArgument("headless", default_value="false")
    timesteps = DeclareLaunchArgument("timesteps", default_value="200000")
    log_root = DeclareLaunchArgument("log_root", default_value=str(workspace_root / "logs" / "ppo_runs"))
    start_moveit = DeclareLaunchArgument("start_moveit", default_value="false")
    robot_description_path = DeclareLaunchArgument(
        "robot_description_path",
        default_value=str(workspace_root / "src" / "robot_description"),
    )

    move_group_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(moveit_launch)),
        condition=IfCondition(LaunchConfiguration("start_moveit")),
    )

    # Default training is Isaac-native and does not depend on ROS 2 Python. MoveIt can still
    # be launched separately for optional integration or debugging, but it is not part of the
    # PPO inner loop.
    trainer = ExecuteProcess(
        cmd=[
            str(isaac_python),
            str(trainer_script),
            "--headless",
            LaunchConfiguration("headless"),
            "--timesteps",
            LaunchConfiguration("timesteps"),
            "--log-root",
            LaunchConfiguration("log_root"),
            "--workspace-root",
            str(workspace_root),
            "--robot-description-path",
            LaunchConfiguration("robot_description_path"),
        ],
        output="screen",
        additional_env={
            "PYTHONPATH": ":".join(pythonpath_entries),
            "HUMANOID_WS_ROOT": str(workspace_root),
        },
    )

    return LaunchDescription(
        [
            isaac_headless,
            timesteps,
            log_root,
            start_moveit,
            robot_description_path,
            move_group_launch,
            trainer,
        ]
    )
