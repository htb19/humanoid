# Robot Readme
除**robot_description包**外，添加了**robot_commander包**，该包用于启动roscontrol，订阅相关话题，方便后续发布话题进行控制。
**robot_moveit_config包**其中是有关moveit的相关配置，当需要修改配置时，可启动该包的ros_moveit插件，修改该文件并将必要文件复制到robot_description包上。
**robot_interfaces包**其中包含自定义的msg文件，方便后续添加，包括位姿话题的msg文件

# 环境配置
~~~
  sudo apt install ros-$ROS_DISTRO-rmw-cyclonedds-cpp
  gedit ~/.bashrc
  export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
~~~
**colcon自动补全**
~~~
  source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash
~~~
**安装moveit**
~~~
  sudo apt install ros-humble-moveit
 ~~~
 
**安装ros_control**
~~~
  sudo apt install ros-humble-ros2-control ros-humble-ros2-controllers
~~~

# Robot Commander
```xml
ros2 launch robot_commander robot_moveit.launch.xml //启动robot_moveit
ros2 run robot_commander commander //启动订阅的相关话题
```
**commander节点还未放入launch文件中，测试是否存在问题**
## 相关话题格式
### 控制夹爪开闭
```
ros2 topic pub -1 /open_left_gripper example_interfaces/msg/Bool "{data: true}"
//左手末端执行器张开
ros2 topic pub -1 /open_left_gripper example_interfaces/msg/Bool "{data: false}"
//左手末端执行器闭合
ros2 topic pub -1 /open_right_gripper example_interfaces/msg/Bool "{data: true}"
//右手末端执行器张开
ros2 topic pub -1 /open_right_gripper example_interfaces/msg/Bool "{data: false}"
//右手手末端执行器闭合
```
### 关节控制
```
ros2 topic pub -1 /right_joint_command example_interfaces/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0, 0.0, 0.2, 0.0]}"\\右手关节控制

ros2 topic pub -1 /left_joint_command example_interfaces/msg/Float64MultiArray "{data: [0.0, 0.0, 0.0, 0.0, 0.2, 0.0]}"\\左手关节控制

ros2 topic pub -1 /neck_joint_command example_interfaces/msg/Float64MultiArray "{data: [0.0, 0.2]}"//颈部控制
```
### 位姿控制
```
ros2 topic pub -1 /left_pose_command robot_interfaces/msg/PoseCommand "{x: 0.0, y: 0.0, z: 0.05, roll: 0.0, pitch: 0.0, yaw: 0.0, cartesian_path: false, relative: false}"
```
**位姿控制存在问题，六轴位姿似乎受限，很多位置无法到达，控制需要修改。**

# Robot Description
## config
此处添加了ros_moveit的配置文件，方便后续各个包的调用。
## urdf
设置humanoid.urdf.xacro文件，方便将urdf文件，ros_control配置文件等进行整合。后续所有的文件的添加应在该文件下整合，方便后续不同模块的修改。
```xml
<?xml version="1.0"?>
<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="humanoid">
    <xacro:arg name="initial_positions_file" default="$(find robot_description)/config/initial_positions.yaml" />
    <xacro:arg name="use_gazebo" default="false"/>

    <!-- Import humanoid urdf file -->
    <xacro:include filename="humanoid.urdf" />
    
    <xacro:include filename="camera_optical_link.xacro"/>
    <xacro:camera_optical_link/>

    <xacro:if value="$(arg use_gazebo)">
      <!-- gazebo 仿真 -->
      <xacro:include filename="humanoid.gazebo.ros2_control.xacro" />
      <xacro:humanoid_gazebo_ros2_control name="GazeboSystem" initial_positions_file="$(arg initial_positions_file)"/>
      
      <!-- 包含 Gazebo 配置 -->
      <xacro:include filename="humanoid.gazebo.xacro"/>
      <xacro:robot_gazebo/>
      
    </xacro:if>
    
    <xacro:unless value="$(arg use_gazebo)">
      <!-- 实机 -->
      
      <!-- Import control_xacro -->
      <xacro:include filename="humanoid.ros2_control.xacro" />
      
      <xacro:humanoid_ros2_control name="FakeSystem" initial_positions_file="$(arg initial_positions_file)"/>
    </xacro:unless>

</robot>
```

* humanoid.urdf包含sw导出的机器人模型文件，
* camera_optical_link.xacro为相机坐标系修改文件
* humanoid.ros2_control.xacro为ros2_control的配置文件
* humanoid.gazebo.ros2_control.xacro为gazebo的仿真文件
* humanoid.gazebo.xacro为gazebo的配置文件

# Robot Moveit Config
插件生成文件，保留方便后续直接在该文件上使用插件进行部分修改。
```
ros2 launch robot_moveit_config demo.launch.py 
//调试机器人运动，测试moveit是否可以正常规划运行
```

# Robot Interfaces
自定义消息文件，后续可以继续配置自定义的msg,srv和action文件。
