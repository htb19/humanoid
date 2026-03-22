from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
import yaml

def load_config(context):
    """从 YAML 文件加载配置字典"""
    config_file = LaunchConfiguration('config_file').perform(context)
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def launch_setup(context):
    """动态加载配置并包含子 launch 文件"""
    config = load_config(context)

    # 包含 aruco_detection.launch.py，传入 aruco_detection 部分的参数
    aruco_detection_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('camera_calibration'),
                'launch',
                'aruco_detection.launch.py'
            ])
        ),
        launch_arguments=config.get('aruco_detection', {}).items()
    )

    # 包含 pose_publisher.launch.py
    end_pose_publisher_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('camera_calibration'),
                'launch',
                'end_pose_publisher.launch.py'
            ])
        ),
        launch_arguments=config.get('pose_publisher', {}).items()
    )

    # 包含 handeye_calibration.launch.py
    handeye_calibration_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('camera_calibration'),
                'launch',
                'handeye_calibration.launch.py'
            ])
        ),
        launch_arguments=config.get('handeye_calibration', {}).items()
    )

    return [
        aruco_detection_launch,
        end_pose_publisher_launch,
        handeye_calibration_launch,
    ]

def generate_launch_description():
    # 声明配置文件路径参数，默认指向 camera_calibration/config/head_cam_handeye_calibration.yaml
    declare_config_file_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('camera_calibration'),
            'config',
            'head_cam_handeye_calibration.yaml'
        ]),
        description='Path to the YAML configuration file'
    )

    return LaunchDescription([
        declare_config_file_arg,
        OpaqueFunction(function=launch_setup)
    ])
