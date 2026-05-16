from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():
    tcp_endpoint_node = Node(
        package="ros_tcp_endpoint",
        executable="default_server_endpoint",
        emulate_tty=True,
        parameters=[{"ROS_IP": "0.0.0.0"}, {"ROS_TCP_PORT": 10000}],
    )

    twist_bridge_node = Node(
        package="robot_servo_control",
        executable="twist_stamp_bridge",
        output="screen",
    )

    twist_bridge_after_endpoint = RegisterEventHandler(
        OnProcessStart(
            target_action=tcp_endpoint_node,
            on_start=[twist_bridge_node],
        )
    )

    return LaunchDescription([
        tcp_endpoint_node,
        twist_bridge_after_endpoint,
    ])
