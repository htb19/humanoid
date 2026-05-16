# ROS-TCP Unity ↔ ROS2 安装与配置说明文档

---

## 一、系统架构说明

本项目采用以下通信架构：

Unity（Windows） ↔ TCP ↔ ROS2（Ubuntu）

- Unity 负责可视化与交互（XR控制）
- ROS2 负责机器人控制（MoveIt Servo）
- 两者通过 ROS-TCP-Connector 进行通信

---

## 二、环境说明

| 组件 | 系统 |
|------|------|
| Unity | Windows |
| ROS2 | Ubuntu |
| 通信方式 | TCP（端口10000） |

---

## 三、Unity 端安装（ROS-TCP-Connector）

### 1. 下载插件

从 GitHub 下载：
https://github.com/Unity-Technologies/ROS-TCP-Connector

点击：Code → Download ZIP

---

### 2. 解压并放入项目

找到文件夹中的：

com.unity.robotics.ros-tcp-connector

放入：

Unity项目/Packages/

确保目录结构如下：

Packages/
 ├── manifest.json
 ├── com.unity.robotics.ros-tcp-connector/
 │    ├── package.json
 │    ├── Runtime/
 │    └── Editor/

---

### 3. 修改 manifest.json

打开：

Packages/manifest.json

在 dependencies 中添加：

"com.unity.robotics.ros-tcp-connector": "file:com.unity.robotics.ros-tcp-connector"

注意：上一行必须有逗号

---

### 4. 重启 Unity

成功标志：

菜单栏出现：Robotics

---

## 四、ROS 端安装（ROS-TCP-Endpoint）

### 1. 下载源码

cd ~/ros2_ws/src

git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git

git checkout -b main-ros2 origin/main-ros2

---

### 2. 编译

cd ~/ros2_ws

colcon build

---

### 3. 加载环境

source install/setup.bash

---

### 4. 启动服务

ros2 launch ros_tcp_endpoint endpoint.launch.py

成功标志：

TCP server listening on 0.0.0.0:10000

---

## 五、Unity 与 ROS 连接配置

### 1. 打开设置

Robotics → ROS Settings

---

### 2. 配置参数

在虚拟机 Ubuntu 内运行：
```bash
hostname -I
```
通常 NAT 模式下 IP 格式为：
VMware：192.168.x.x 或 172.16.x.x

| 参数 | 值 |
|------|----|
| ROS IP | 192.168.x.x |
| Port | 10000 |

---

### 3. 建立连接

点击 Connect

成功标志：Connected to ROS

---

## 六、ROS 消息生成（关键步骤）

### 问题背景

Unity 无法直接访问 Ubuntu 中的 ROS 消息文件（geometry_msgs、std_msgs）

---

### 解决方案（推荐）

#### 方法：复制 ROS 消息到 Windows

1. 在 Ubuntu 执行：

cd /opt/ros/humble/share

zip -r ros_msgs.zip geometry_msgs std_msgs

2. 将压缩包复制到 Windows，例如：

C:\ros_msgs\

3. 解压后目录应为：

C:\ros_msgs\
 ├── geometry_msgs
 └── std_msgs

---

### Unity 配置

打开：

Robotics → ROS Settings

设置：

ROS Package Path = C:\ros_msgs

---


## 七、通信测试

### ROS 发布测试

ros2 topic pub /test std_msgs/msg/String "{data: 'hello'}"

---

### Unity 订阅测试

编写脚本：

Subscribe /test 并输出日志

---

## 八、常见问题

### 1. 报错：package.json 找不到

原因：路径错误（多写了 Packages/）

解决：

"file:com.unity.robotics.ros-tcp-connector"

---

### 2. 无法选择 geometry_msgs

原因：Unity 找不到 ROS msg 文件

解决：复制到 Windows 并设置 ROS Package Path

---

### 3. Unity 无法连接 ROS

检查：

- IP 是否正确
- 端口是否 10000
- ROS 端是否启动 endpoint

---

### 4. Unity 没有 Robotics 菜单

原因：插件未正确安装

---

## 九、总结

完成以下步骤即可实现 Unity ↔ ROS 通信：

1. 安装 ROS-TCP-Connector（Unity）
2. 安装 ROS-TCP-Endpoint（ROS）
3. 配置 TCP 连接
4. 配置 ROS Package Path
5. 生成消息

---

## 十、后续扩展

- Unity → MoveIt Servo 控制
- XR 双手控制机器人
- Unity 可视化机器人状态

---

（文档完）

