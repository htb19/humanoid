from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # aruco_ros single 节点
    """
    aruco_single 节点默认发布：
      /aruco_single/pose（PoseStamped）
      /aruco_single/result（Image）
    """
    aruco_node = Node(
        package='aruco_ros',
        executable='single',
        name='aruco_single',
        output='screen',
        parameters=[{
            'marker_id': 582,
            'marker_size': 0.2,
            'image_is_rectified': False,
            'reference_frame': '',
            'camera_frame': 'head_camera_optical_link',
            'marker_frame': 'aruco_marker',  # 发布的 TF 子坐标系
            'corner_refinement': 'LINES',
        }],
        remappings=[
            ('/camera_info', '/head_camera/camera_info'),
            ('/image', '/head_camera/image'),
        ]
    )
    
    return LaunchDescription([
        aruco_node,
    ])
