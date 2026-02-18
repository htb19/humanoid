from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    保存数据：
   ros2 service call /save_handeye_pair std_srvs/srv/Trigger

    计算并发布结果：
   ros2 service call /compute_handeye std_srvs/srv/Trigger
    """
    return LaunchDescription([
        Node(
            package='robot_simulation',
            executable='handeye_calibration',
            name='handeye_calibration',
            output='screen',
            parameters=[{
                'camera_pose_topic': '/aruco_single/pose',
                'gripper_pose_topic': '/end_effector_pose',
                'gripper_frame': 'head_camera_link',
                'camera_frame': 'head_camera_optical_link',
                'output_pose_topic': '/cam2gripper_pose'
            }]
        )
    ])
