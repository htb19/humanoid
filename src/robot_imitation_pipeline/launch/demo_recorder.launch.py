from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    config_arg = DeclareLaunchArgument(
        "config",
        default_value=PathJoinSubstitution(
            [FindPackageShare("robot_imitation_pipeline"), "config", "recording.yaml"]
        ),
        description="Recorder YAML config.",
    )

    recorder = Node(
        package="robot_imitation_pipeline",
        executable="demo_recorder_node",
        name="demo_recorder",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )

    return LaunchDescription([config_arg, recorder])
