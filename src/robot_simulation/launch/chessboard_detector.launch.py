from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_simulation',
            executable='chessboard_detector',
            name='chessboard_detector',
            output='screen',
            parameters=[{
                'image_topic': '/head_camera/image',
                'camera_info_topic': '/head_camera/camera_info',
                'pattern_cols': 7,
                'pattern_rows': 7,
                'square_size': 0.0625,# 单位：米
                'camera_frame': 'head_camera_optical_link',
                'board_frame': 'chessboard',
                'pose_topic': '/chessboard_pose'
            }]
        )
    ])
