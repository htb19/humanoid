# robot_workspace_quality_analyzer

独立的 MoveIt 工作空间质量分析包，用于在当前末端位姿附近采样候选点，评估 Servo 运动稳定性相关风险：

- IK 是否可达
- Jacobian 最小奇异值
- Jacobian 条件数
- manipulability
- 关节限位 margin
- 碰撞状态
- IK 解距离当前关节状态的距离
- RViz MarkerArray 可视化
- CSV 输出

这个包不修改 Servo 控制链路，也不修改已有机器人包。它只读取 MoveIt 当前状态和 PlanningScene。

## 构建

在工作区根目录执行：

```bash
colcon build --packages-select robot_workspace_quality_analyzer
source install/setup.bash
```

## 启动

需要先启动你的机器人描述、MoveIt move_group、joint_states 或仿真，使 `robot_description`、当前关节状态、PlanningScene 可用。

```bash
ros2 launch robot_workspace_quality_analyzer workspace_quality_analyzer.launch.py
```

默认规划组设置为当前 `robot_moveit_config` 中的 `right_arm`。如果你要分析左臂或双臂，请修改：

```text
config/workspace_quality_analyzer.yaml
```

重点参数：

```yaml
move_group_name: "right_arm"
ee_link: ""
base_frame: "base_link"
resolution: 0.05
csv_output_dir: "/tmp/workspace_quality"
```

`ee_link` 为空时，节点会使用 MoveIt 中该 planning group 的末端 link。

## 触发分析

节点不会持续高频计算。启动后调用服务触发一次分析：

```bash
ros2 service call /workspace_quality_analyzer/analyze std_srvs/srv/Trigger {}
```

输出：

- RViz MarkerArray topic: `/workspace_quality/markers`
- 最优点 PoseStamped topic: `/workspace_quality/best_pose`
- CSV 文件：`csv_output_dir` 下的 `workspace_quality_*.csv`

## RViz 颜色

- 绿色：质量较好
- 黄色：警告
- 红色：奇异、关节限位或距离当前状态过远
- 灰色：IK 不可达
- 紫色：碰撞
- 青色：当前 TCP
- 白色：最佳候选点

## CSV 字段

```csv
index,x,y,z,qx,qy,qz,qw,ik_success,collision,score,risk_level,risk_reason,sigma_min,condition_number,manipulability,joint_margin_min,q_distance,ik_time_ms,joint_values
```

## 建议

第一轮建议使用较粗分辨率，例如 `0.05` 米。确认 IK 和 RViz 显示正常后，再降低到 `0.03` 米或加入姿态扰动采样。
