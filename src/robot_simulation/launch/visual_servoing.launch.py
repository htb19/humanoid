from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('robot_simulation'),
                'launch',
                'gazebo.launch.py'
            ])
        ),
        launch_arguments={
            'config_file': 'visual_servoing_sim_config.yaml'
        }.items()
    )

    return LaunchDescription([
        gazebo_launch,
    ])
    
    
