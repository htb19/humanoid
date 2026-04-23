import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except OSError:
        return None


def servo_node(namespace, yaml_file, moveit_config):
    servo_yaml = load_yaml("robot_servo_control", yaml_file)
    return Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="moveit_servo",
        namespace=namespace,
        parameters=[
            {"moveit_servo": servo_yaml},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
        ],
        remappings=[
            ("get_planning_scene", "/get_planning_scene"),
            ("apply_planning_scene", "/apply_planning_scene"),
        ],
        output="screen",
    )


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("robot")
        .robot_description(file_path="config/humanoid.urdf.xacro")
        .to_moveit_configs()
    )

    head_servo_node = servo_node(
        "head_servo",
        "config/head_servo.yaml",
        moveit_config,
    )
    left_servo_node = servo_node(
        "left_servo",
        "config/left_arm_servo.yaml",
        moveit_config,
    )
    right_servo_node = servo_node(
        "right_servo",
        "config/right_arm_servo.yaml",
        moveit_config,
    )

    return LaunchDescription(
        [
            head_servo_node,
            left_servo_node,
            right_servo_node,
        ]
    )
