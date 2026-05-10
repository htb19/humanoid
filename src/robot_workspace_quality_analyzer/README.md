# robot_workspace_quality_analyzer

独立的 MoveIt 工作空间质量分析包，用于在当前末端位姿附近采样候选点，评估 Servo 运动稳定性相关风险：

- IK 是否可达
- Jacobian 最小奇异值
- Jacobian 条件数
- manipulability
- 关节限位 margin
- 碰撞状态
- IK 解距离当前关节状态的距离
- **SO(3) 姿态均匀采样与优化**
- RViz MarkerArray 可视化（位置点云 + 姿态球壳）
- CSV 输出
- **移动到最佳位姿（四阶段工作流）**

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
sample_shape: "sphere"
samples_per_dim: 11
so3_sample_count: 125
max_orientation_samples: 1000
orientation_range_deg: 30.0
orientation_scan_timeout: 30.0
orientation_shell_radius: 0.08
orientation_visual_axis: "z"
write_csv: true
marker_scale: 1.0
csv_output_dir: "/tmp/workspace_quality"
```

`ee_link` 为空时，节点会使用 MoveIt 中该 planning group 的末端 link。

`sample_shape` 为 `"sphere"` 时，采样空间为以当前末端位姿为中心的实心球体。`samples_per_dim` 控制包围球体的立方体网格密度，分辨率自动计算。

姿态扫描参数：
- `so3_sample_count`：SO(3) 姿态采样总数（非 Euler 网格），内部自动分解为 S² 方向数 × SO(2) 旋转数
- `max_orientation_samples`：硬上限保护，防止误设过大导致卡死
- `orientation_range_deg`：姿态搜索半范围(度)，默认 90，≥180 为全 SO(3) 搜索
- `orientation_scan_timeout`：超时保护(秒)，超时返回已计算结果
- `orientation_shell_radius`：RViz 姿态球壳半径(m)
- `orientation_visual_axis`：工具指向轴（`"x"` / `"y"` / `"z"`），不同机械臂可能不同
- `write_csv`：是否输出 CSV 文件（`true`/`false`），调试时可关闭避免垃圾文件
- `marker_scale`：小球大小缩放因子，默认 1.0。自适应计算基础大小（位置=半径/密度×1.2，姿态=壳半径/∛N×1.5），此参数统一缩放

## 服务

### /analyze

触发工作空间质量分析：

```bash
ros2 service call /workspace_quality_analyzer/analyze robot_workspace_quality_analyzer/srv/AnalyzeWorkspace "{sphere_radius: 0.15, samples_per_dim: 11}"
```

参数（不传则用 config 默认值）：
- `sphere_radius`：采样球半径(m)
- `samples_per_dim`：每维采样密度

输出：
- RViz MarkerArray topic: `/workspace_quality/markers`
- 最优点 PoseStamped topic: `/workspace_quality/best_pose`
- CSV 文件：`csv_output_dir` 下的 `workspace_quality_*.csv`
- 保存最佳候选结果供 `/move_to_best` 使用（`source=POSITION_SCAN`）

### /analyze_orientation

在当前位置对末端姿态进行 SO(3) 均匀采样分析（S² × SO(2) 分解），找到兼顾安全性和姿态接近度的最优姿态：

```bash
ros2 service call /workspace_quality_analyzer/analyze_orientation robot_workspace_quality_analyzer/srv/AnalyzeOrientation "{so3_sample_count: 125, orientation_range_deg: 90.0}"
```

参数（不传则用 config 默认值）：
- `so3_sample_count`：SO(3) 姿态采样总数
- `orientation_range_deg`：姿态搜索半范围(度)，默认 90

输出：
- RViz 姿态球壳：`workspace_quality_orientation_shell` namespace，颜色点表示质量
- CSV 文件新增 `orientation_error_rad`、`source` 列
- 覆盖 `best_result_`，供 `/move_to_best` 复用（`source=ORIENTATION_SCAN`）

选择逻辑（分层过滤）：
1. 安全门槛：仅保留 GOOD / WARNING
2. 质量排序：按 `risk_level` → `score` → `orientation_error_rad`

### /move_to_best

规划并执行移动到上一次分析得到的最佳关节构型。必须先调用 `/analyze` 获得有效结果。

```bash
ros2 service call /workspace_quality_analyzer/move_to_best std_srvs/srv/Trigger {}
```

安全特性：

- 使用关节空间目标（`setJointValueTarget`），避免二次 IK 不确定性
- 规划前同步当前状态（`setStartStateToCurrentState`）
- 速度缩放 0.2、加速度缩放 0.2，避免真实机器人运动过快
- 校验 joint_values 维度与合法性
- 未分析或分析结果无效时返回错误
- 线程安全：`std::mutex` 保护最佳结果读写
- 响应携带 `source=POSITION_SCAN|ORIENTATION_SCAN` 区分最佳位姿来源

### 典型工作流（四阶段）

```bash
# Stage 1: 位置扫描
ros2 service call /workspace_quality_analyzer/analyze robot_workspace_quality_analyzer/srv/AnalyzeWorkspace "{sphere_radius: 0.15, samples_per_dim: 9}"

# Stage 2: 移动到最佳位置（在 RViz 中查看 MarkerArray 确认）
ros2 service call /workspace_quality_analyzer/move_to_best std_srvs/srv/Trigger {}

# Stage 3: 姿态扫描（在当前位置）
ros2 service call /workspace_quality_analyzer/analyze_orientation robot_workspace_quality_analyzer/srv/AnalyzeOrientation "{so3_sample_count: 512, orientation_range_deg: 90.0}"

# Stage 4: 移动到最佳姿态（复用 /move_to_best）
ros2 service call /workspace_quality_analyzer/move_to_best std_srvs/srv/Trigger {}
```

## RViz 颜色

- 绿色：质量较好
- 黄色：警告
- 红色：奇异、关节限位或距离当前状态过远
- 灰色：IK 不可达
- 紫色：碰撞
- 青色：当前 TCP
- 白色：最佳候选点（位置扫描在 3D 空间，姿态扫描投影到球壳）

姿态扫描 RViz namespace：
- `workspace_quality_orientation_shell`：方向球壳点（SPHERE），颜色表示质量，大小自适应（`marker_scale` 统一缩放）
- `workspace_quality_best`：最佳姿态候选（白色大球，与位置扫描共用）

## CSV 字段

```csv
index,x,y,z,qx,qy,qz,qw,ik_success,collision,score,risk_level,risk_reason,sigma_min,condition_number,manipulability,joint_margin_min,q_distance,orientation_error_rad,source,ik_time_ms,joint_values
```

- `orientation_error_rad`：与参考姿态的角距离（位置扫描为 0）
- `source`：`position` 或 `orientation`

## 线程模型

节点使用 `MultiThreadedExecutor`，避免 MoveIt 内部 action/TF/state monitor 在单线程下死锁。`/analyze`、`/analyze_orientation` 和 `/move_to_best` 可被不同线程并发调用，`best_result_` 由 `std::mutex` 保护读写安全。

## MoveIt 初始化说明

`MoveGroupInterface` 在 `initializeMoveIt()` 中创建（于 `main()` 的 `make_shared` 之后），此时 `shared_from_this()` 安全。初始化时启动 StateMonitor 并等待首次状态获取（超时 5 秒），确保规划前状态可用。

## 建议

第一轮建议使用较大半径（如 `0.25`）快速粗扫确认功能正常。确认 IK 和 RViz 显示正常后，再缩小半径做精细分析。如需调整采样密度，修改配置文件中的 `samples_per_dim`（默认 11，球体约 700 个采样点）。

姿态扫描默认 90° 半球搜索，`so3_sample_count` 默认 125（25 方向 × 5 旋转），足够均匀覆盖。调试时可减小采样数加快速度，精细搜索可增加到 512 或更高。全 SO(3) 搜索设置 `orientation_range_deg: 180.0`。
