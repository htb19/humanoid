from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_simulation',
            executable='end_effector_pose_publisher',
            name='end_effector_pose_publisher',
            output='screen',
            parameters=[{
                'source_frame': 'base_link',
                'target_frame': 'head_camera_link',
                'pose_topic': '/end_effector_pose',
                'publish_rate': 20.0
            }]
        )
    ])
