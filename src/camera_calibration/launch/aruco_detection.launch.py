from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # 声明参数，并设置默认值
    marker_id_arg = DeclareLaunchArgument(
        'marker_id',
        default_value='582',
        description='ArUco marker ID'
    )
    marker_size_arg = DeclareLaunchArgument(
        'marker_size',
        default_value='0.2',
        description='ArUco marker size in meters'
    )
    image_is_rectified_arg = DeclareLaunchArgument(
        'image_is_rectified',
        default_value='false',
        description='Whether the image is rectified'
    )
    reference_frame_arg = DeclareLaunchArgument(
        'reference_frame',
        default_value='',
        description='Reference frame for pose (empty means camera frame)'
    )
    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='head_camera_optical_link',
        description='Camera optical frame'
    )
    marker_frame_arg = DeclareLaunchArgument(
        'marker_frame',
        default_value='aruco_marker',
        description='TF child frame for the marker'
    )
    corner_refinement_arg = DeclareLaunchArgument(
        'corner_refinement',
        default_value='LINES',
        description='Corner refinement method (NONE, LINES, SUBPIX)'
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/head_camera/camera_info',
        description='Camera info topic'
    )
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/head_camera/image',
        description='Image topic'
    )

    
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
            'marker_id': LaunchConfiguration('marker_id'),
            'marker_size': LaunchConfiguration('marker_size'),
            'image_is_rectified': LaunchConfiguration('image_is_rectified'),
            'reference_frame': LaunchConfiguration('reference_frame'),
            'camera_frame': LaunchConfiguration('camera_frame'),
            'marker_frame': LaunchConfiguration('marker_frame'),  # 发布的 TF 子坐标系
            'corner_refinement': LaunchConfiguration('corner_refinement'),
        }],
        remappings=[
            ('/camera_info', LaunchConfiguration('camera_info_topic')),
            ('/image', LaunchConfiguration('image_topic')),
        ]
    )
    
    return LaunchDescription([
        marker_id_arg,
        marker_size_arg,
        image_is_rectified_arg,
        reference_frame_arg,
        camera_frame_arg,
        marker_frame_arg,
        corner_refinement_arg,
        camera_info_topic_arg,
        image_topic_arg,
        aruco_node,
    ])
