from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    
    # 声明参数
    camera_pose_topic_arg = DeclareLaunchArgument(
        'camera_pose_topic',
        default_value='/aruco_single/pose',
        description='Topic for camera to marker pose'
    )
    gripper_pose_topic_arg = DeclareLaunchArgument(
        'gripper_pose_topic',
        default_value='/end_effector_pose',
        description='Topic for gripper pose in base frame'
    )
    gripper_frame_arg = DeclareLaunchArgument(
        'gripper_frame',
        default_value='head_camera_link',
        description='Gripper frame (parent of output tf)'
    )
    camera_frame_arg = DeclareLaunchArgument(
        'camera_frame',
        default_value='head_camera_optical_link',
        description='Camera frame (child of output tf)'
    )
    output_pose_topic_arg = DeclareLaunchArgument(
        'output_pose_topic',
        default_value='/cam2gripper_pose',
        description='Topic for output calibration result'
    )

    """
    handeye_calibration节点：
    #保存数据：
    ros2 service call /save_handeye_pair std_srvs/srv/Trigger

    #计算并发布结果：
    ros2 service call /compute_handeye std_srvs/srv/Trigger
    """
    handeye_calibration = Node(
        package='camera_calibration',
        executable='handeye_calibration',
        name='handeye_calibration',
        output='screen',
        parameters=[{
            'camera_pose_topic': LaunchConfiguration('camera_pose_topic'),
            'gripper_pose_topic': LaunchConfiguration('gripper_pose_topic'),
            'gripper_frame': LaunchConfiguration('gripper_frame'),
            'camera_frame': LaunchConfiguration('camera_frame'),
            'output_pose_topic': LaunchConfiguration('output_pose_topic'),
        }]
    )
    
    return LaunchDescription([
        camera_pose_topic_arg,
        gripper_pose_topic_arg,
        gripper_frame_arg,
        camera_frame_arg,
        output_pose_topic_arg,
        handeye_calibration
    ])
