import os
import yaml
from launch import LaunchDescription
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from moveit_configs_utils import MoveItConfigsBuilder


def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, "r") as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None


def generate_launch_description():

    moveit_config = (
        MoveItConfigsBuilder("robot")
        .robot_description(file_path="config/humanoid.urdf.xacro")
        .to_moveit_configs()
    )

    # ── 左臂 Servo ──────────────────────────────────────────
    left_servo_yaml = load_yaml("robot_servo_control", "config/left_arm_servo.yaml")
    left_servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="moveit_servo",
        namespace="left_servo",
        parameters=[
            {"moveit_servo": left_servo_yaml},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            # {"use_sim_time": True}，
        ],
        
        remappings=[
            ("get_planning_scene", "/get_planning_scene"),
            ("apply_planning_scene", "/apply_planning_scene"),
        ],        
        output="screen",
    )

    # ── 右臂 Servo ──────────────────────────────────────────
    right_servo_yaml = load_yaml("robot_servo_control", "config/right_arm_servo.yaml")
    right_servo_node = Node(
        package="moveit_servo",
        executable="servo_node_main",
        name="moveit_servo",
        namespace="right_servo",
        parameters=[
            {"moveit_servo": right_servo_yaml},
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            # {"use_sim_time": True},
        ],

        remappings=[
            ("get_planning_scene", "/get_planning_scene"),
            ("apply_planning_scene", "/apply_planning_scene"),
        ],
        output="screen",
    )

    return LaunchDescription([
        left_servo_node,
        right_servo_node,
    ])