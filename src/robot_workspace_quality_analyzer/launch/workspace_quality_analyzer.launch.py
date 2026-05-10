from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    moveit_config = (
        MoveItConfigsBuilder("humanoid", package_name="robot_moveit_config")
        .to_moveit_configs()
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("robot_workspace_quality_analyzer"),
                        "config",
                        "workspace_quality_analyzer.yaml",
                    ]
                ),
            ),
            Node(
                package="robot_workspace_quality_analyzer",
                executable="workspace_quality_analyzer_node",
                name="workspace_quality_analyzer",
                output="screen",
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    config_file,
                ],
            ),
        ]
    )
