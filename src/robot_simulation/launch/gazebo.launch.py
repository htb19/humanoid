# gazebo.launch.py
from launch import LaunchDescription, LaunchContext
from launch.actions import (
    SetEnvironmentVariable,
    IncludeLaunchDescription,
    ExecuteProcess,
    RegisterEventHandler,
    DeclareLaunchArgument,
    OpaqueFunction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    PathJoinSubstitution,
    Command,
    LaunchConfiguration,
)
from launch_ros.actions import Node, SetParameter
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os, yaml
from moveit_configs_utils import MoveItConfigsBuilder

def collect_enabled(config, prefix=''):
    """扁平化收集所有启用的节点名（自动处理all_disable）"""
    if config.get('all_disable', False):
        return []
    enabled = []
    for k, v in config.items():
        if k == 'all_disable':
            continue
        if isinstance(v, dict):
            enabled.extend(collect_enabled(v, prefix + k + '.'))
        elif v is True:
            enabled.append(k)
    return enabled

def launch_setup(context: LaunchContext):
    
    # ============ 1. 路径声明 ============
    # /install/robot_simulation/share/robot_simulation
    pkg_share = FindPackageShare("robot_simulation")
    
    # 加载配置文件
    config_file = LaunchConfiguration('config_file').perform(context)
    
    if os.path.isabs(config_file):
        config_path = config_file
    else:
        config_path = os.path.join(
            get_package_share_directory("robot_simulation"),
            "config",
            config_file
        )
    print(f"Config path: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置 Gazebo 资源路径
    set_resource_path = SetEnvironmentVariable(
        name = 'IGN_GAZEBO_RESOURCE_PATH',
        value = [
            PathJoinSubstitution([
                FindPackageShare("robot_description"), ".."
            ]),
            ':',
            PathJoinSubstitution([pkg_share, "models"])
        ]
    )
    
    # 机器人描述文件路径（URDF）
    xacro_path = PathJoinSubstitution([
        FindPackageShare("robot_description"),
        "urdf",
        "humanoid.urdf.xacro"
    ])
    
    world_file = PathJoinSubstitution([
        pkg_share,
        config['simulation']['world_file']
    ])
    
    moveit_config = MoveItConfigsBuilder(
        "humanoid",
        package_name="robot_moveit_config"
    ).to_moveit_configs()
    
    rviz2_config = PathJoinSubstitution([
        pkg_share,
        "rviz",
        "simulation.rviz"
    ])
    
    rviz2_parameters = [
        moveit_config.planning_pipelines,
        moveit_config.robot_description_kinematics,
        moveit_config.joint_limits,
    ]
    
    # ============ 2. 核心节点定义 ============
    # (1) 启动空世界Gazebo Sim
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py"
            ])
        ]),
        launch_arguments={
            "gz_args": [
                "-r -v 1 ",
                world_file
            ]
        }.items()
    )
    
    # (2) 机器人状态发布器（发布TF树，供RVIZ使用）
    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": Command([
                'xacro ', xacro_path, " use_gazebo:=true"]),
            # "use_sim_time": True,
        }]
    )
    
    # (3) 模型生成器
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name", "humanoid",
            "-topic", "/robot_description",
            "-z", "0.1",  # 抬高1米避免穿模
            "-x", "0",
            "-y", "0"
        ],
        output="screen"
    )
    
    # ============ 3. 加载控制器 ============
    # 关节状态广播器（基础，必须最先加载）
    joint_state_broadcaster = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster", "--controller-manager", "/controller_manager"],
        output="screen"
    )
    
    load_joint_state_broadcaster = RegisterEventHandler(
        event_handler = OnProcessExit(
            target_action = spawn_entity,
            on_exit = [joint_state_broadcaster],
        )
    )
    
    # 头部控制器
    neck_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["neck_controller", "--controller-manager", "/controller_manager"],
        output="screen"
    )
    
    # 右臂控制器
    right_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_arm_controller", "--controller-manager", "/controller_manager"],
        output="screen"
    )
    
    # 右夹爪控制器
    right_gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["right_gripper_controller", "--controller-manager", "/controller_manager"],
        output="screen"
    )
    
    # 左臂控制器
    left_arm_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_arm_controller", "--controller-manager", "/controller_manager"],
        output="screen"
    )
    
    # 左夹爪控制器
    left_gripper_controller = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["left_gripper_controller", "--controller-manager", "/controller_manager"],
        output="screen"
    )
    
    # 桥接话题
    topic_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            # 时钟
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
            # RGB相机：
            '/right_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/right_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/left_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/left_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            
            # 头部realsense相机：
            '/head_camera/camera_info@sensor_msgs/msg/CameraInfo[ignition.msgs.CameraInfo',
            '/head_camera/image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/head_camera/depth_image@sensor_msgs/msg/Image[ignition.msgs.Image',
            '/head_camera/points@sensor_msgs/msg/PointCloud2[ignition.msgs.PointCloudPacked',
        ],
        output='screen'
    )
    
    rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz2_config],
        parameters=rviz2_parameters,
    )
    
    rqt = Node(
        package="rqt_gui",
        executable="rqt_gui",
        output="screen",
        arguments=[
            "--perspective-file",
            PathJoinSubstitution([pkg_share, "config", "rqt.perspective"])
        ],
        parameters=[{"use_sim_time": False}]
    )
    
    moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('robot_moveit_config'),
            '/launch/move_group.launch.py'
        ]),
        # launch_arguments={'use_sim_time': 'true'}.items()
    )
    
    # ============ 4. 启动流程 ============
    # 核心节点
    core_nodes = [
        set_resource_path,
        
        # 核心流程
        gz_sim,
        robot_state_publisher,
        spawn_entity,
        load_joint_state_broadcaster,
    ]
    
    # 可选节点注册
    opts_dict = {
        # yaml配置的key : launch节点
        # ROS2控制器
        'neck_controller': neck_controller,
        'right_arm_controller': right_arm_controller,
        'right_gripper_controller': right_gripper_controller,
        'left_arm_controller': left_arm_controller,
        'left_gripper_controller': left_gripper_controller,
        # 话题桥接
        'topic_bridge': topic_bridge,
        # moveit
        'moveit': moveit,
        # GUI-APP
        'rviz2': rviz2,
        'rqt': rqt,
    }
    
    # 查出启动列表
    options_list = [opts_dict[k] for k in collect_enabled(config['nodes_enable']) if k in opts_dict]
    
    if options_list:
        # 关节状态发布后，启动可选节点
        optional_nodes = [RegisterEventHandler(
            OnProcessExit(
                target_action = joint_state_broadcaster,
                on_exit = options_list
            )
        )]
    else:
        optional_nodes = []
    
    return core_nodes + optional_nodes
 
    
def generate_launch_description(): 
    return LaunchDescription([
        SetParameter(name='use_sim_time', value=True),
        
        # 声明参数
        DeclareLaunchArgument(
            'config_file',
            default_value='default_sim_config.yaml',
            description='Config file name in robot_simulation/config/ or absolute path'
        ),
        
        # 延迟执行以获取参数值
        OpaqueFunction(function=launch_setup)
    ])
    
    
