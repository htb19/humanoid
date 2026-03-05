from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    source_frame_arg = DeclareLaunchArgument(
        'source_frame',
        default_value='base_link',
        description='Source frame for TF lookup'
    )
    target_frame_arg = DeclareLaunchArgument(
        'target_frame',
        default_value='head_camera_link',
        description='Target frame for TF lookup'
    )
    pose_topic_arg = DeclareLaunchArgument(
        'pose_topic',
        default_value='/end_effector_pose',
        description='Topic to publish PoseStamped'
    )
    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='20.0',
        description='Publishing rate in Hz'
    )
    
    pose_pub_node = Node(
        package='camera_calibration',
        executable='end_effector_pose_publisher',
        name='end_effector_pose_publisher',
        output='screen',
        parameters=[{
            'source_frame': LaunchConfiguration('source_frame'),
            'target_frame': LaunchConfiguration('target_frame'),
            'pose_topic': LaunchConfiguration('pose_topic'),
            'publish_rate': LaunchConfiguration('publish_rate'),
        }]
    )
    
    return LaunchDescription([
        source_frame_arg,
        target_frame_arg,
        pose_topic_arg,
        publish_rate_arg,
        pose_pub_node,
    ])
