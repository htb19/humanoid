# Unity 端遥操作设计修改总结

## 目标

这份文档总结当前 Unity 遥操作链路中，建议在 Unity 端完成的设计调整，目标是同时改善两件事：

- 手柄位姿增量更跟手
- 输出给 ROS / MoveIt Servo 的 `TwistStamped` 更稳定、更接近真实机器人可执行的运动意图

当前工程的主要实现脚本是：

- `Assets/Project/Scripts/DualXRServo.cs`

---

## 当前设计概况

当前脚本已经实现了下面这条链路：

- 读取左右手控制器位姿
- 在参考坐标系下计算位置和姿态增量
- 将增量除以 `dt` 转成线速度和角速度
- 通过 `TwistStamped` 发布到 ROS topic

当前版本的优点：

- 已经有 `referenceFrame`，可以消除头动和 XR Origin 整体位移的影响
- 已经有左右手使能、trigger 使能、坐标系转换、死区、限幅等基础能力
- 已经能直接对接 ROS bridge 和 MoveIt Servo

当前版本的主要问题：

- 手柄采样和 ROS 发布都放在 `FixedUpdate()` 中，输入采样不够“贴近 XR 刷新时机”
- 速度平滑使用 `Lerp`，参数较小时会带来明显拖手感
- “手感优化”和“机器人动态约束”耦合在一起，后期不容易分层调试

---

## Unity 端应该负责什么

Unity 端应负责“人手意图生成”，也就是最接近 Quest 2 输入、最影响操作手感的部分。

建议 Unity 负责：

- Quest 2 控制器位姿采样
- 参考坐标系下的相对位姿计算
- 位姿增量转 twist
- 轻死区
- 比例缩放
- 轻量平滑
- 输入使能逻辑
- Unity 到 ROS 的坐标轴转换

不建议 Unity 负责：

- 最终机器人安全边界
- 很重的速度硬限幅
- 很保守的强滤波
- 复杂 watchdog
- 机器人专属工作空间限制

原则是：

- Unity 输出“操作者的运动意图”
- ROS 输出“机器人最终允许执行的运动”

---

## 必须修改的设计点

## 1. 将控制器采样从 `FixedUpdate()` 拆分到 `Update()`

### 现状

当前脚本在 `FixedUpdate()` 中直接读取控制器位姿、计算增量并发布。

### 问题

- `FixedUpdate()` 是物理时钟，不一定与 XR 控制器姿态更新频率一致
- Quest 2 控制器位姿更接近渲染/输入更新节奏
- 在 `FixedUpdate()` 里直接采样，容易出现重复采样、漏采样、细节丢失
- 速度估计会更容易抖动或滞后

### 建议设计

改成两层：

- `Update()`：只负责采样控制器最新 pose，并记录采样时间
- `FixedUpdate()`：只负责用最近一次缓存数据计算/发布控制量

### 预期收益

- 输入更贴近 Quest 2 实际刷新节奏
- 更容易提升“跟手感”
- 后续调试时可以把“输入采样问题”和“输出控制问题”分开

---

## 2. 用“加速度限制”替代当前重 `Lerp` 平滑

### 现状

当前线速度和角速度平滑方式是：

- `smoothVel = Vector3.Lerp(smoothVel, rawLinearVelRef, smoothFactor)`
- `smoothAngularVel = Vector3.Lerp(smoothAngularVel, rawAngularVelRef, smoothFactor)`

### 问题

- `Lerp` 是信号平滑，不是动力学约束
- `smoothFactor` 偏小时，输出会明显追着目标跑
- 会让操作者感到“黏”“拖”“不跟手”

### 建议设计

改成：

- 先算目标线速度 `targetLinear`
- 先算目标角速度 `targetAngular`
- 当前输出速度 `currentLinear`、`currentAngular`
- 每一帧只允许它们最多变化 `maxAccel * dt`

线速度示意：

```csharp
Vector3 delta = targetLinear - currentLinear;
Vector3 limitedDelta = Vector3.ClampMagnitude(delta, maxLinearAccel * dt);
currentLinear += limitedDelta;
```

角速度示意：

```csharp
Vector3 delta = targetAngular - currentAngular;
Vector3 limitedDelta = Vector3.ClampMagnitude(delta, maxAngularAccel * dt);
currentAngular += limitedDelta;
```

### 预期收益

- 小动作更直接
- 大动作不会突然过猛
- 更像真实机器人受动力学限制，而不是信号被抹平

### Inspector 新参数建议

- `maxLinearAccel`
- `maxAngularAccel`

---

## 3. 保留 `referenceFrame`，并明确绑定到 XR Origin

### 现状

当前脚本已经支持 `referenceFrame`。

### 建议设计

`referenceFrame` 明确绑定：

- `XR Origin`
- 或 tracking space root

不要绑定到：

- HMD Camera
- 左右手控制器本身

### 原因

- 可以消除头部移动带来的虚假手部增量
- 可以让手柄控制更像“相对工作空间操作”
- 对双臂遥操作尤其重要

---

## 4. 重新定义 Unity 端参数职责

当前参数混合了手感参数和控制约束参数，建议在 Unity 端重新定义为“意图层参数”。

### Unity 端建议保留的参数

- `linearScale`
- `angularScale`
- `linearDeadZone`
- `angularDeadZone`
- `maxLinearIntent`
- `maxAngularIntent`
- `maxLinearAccel`
- `maxAngularAccel`
- `triggerThreshold`

### Unity 端参数含义建议

- `Scale`：决定手柄位移/转角映射到机器人意图速度的比例
- `DeadZone`：过滤手柄微抖
- `Intent Max`：限制操作者单次输入能表达出的最大意图速度
- `Accel`：保证意图变化不会过于激进，但不过度损失跟手感

### 不建议在 Unity 端定义为最终安全值

- 最终最大线速度
- 最终最大角速度
- 最终最大加速度
- 工作空间边界

这些应在 ROS 侧做硬约束。

---

## 5. 使用真实采样时间计算增量速度，而不是固定依赖 `fixedDeltaTime`

### 现状

当前 `dt` 主要依赖：

- `Time.fixedDeltaTime`
- `minDeltaTime`

### 问题

如果采样改到 `Update()`，就不能继续用固定物理步长近似真实输入采样间隔。

### 建议设计

每次采样时缓存：

- `sampleTime`
- `samplePosition`
- `sampleRotation`

计算速度时用：

- `sampleTime - lastSampleTime`

### 预期收益

- 线速度估计更准确
- 角速度估计更准确
- 更利于 Quest 2 在不同刷新率下保持稳定手感

---

## 6. 将“输入意图生成”和“ROS 发布”在代码结构上分离

### 建议结构

建议在 Unity 脚本中按职责拆为两个阶段：

- 阶段 A：采样与意图生成
- 阶段 B：消息构建与发布

### 推荐内部流程

对于每只手：

1. 读取控制器当前 pose
2. 转换到 `referenceFrame`
3. 与上一采样 pose 求增量
4. 计算目标线速度和目标角速度
5. 应用死区与比例缩放
6. 应用加速度限制
7. 转为 ROS 坐标系
8. 发布 `TwistStamped`

### 这样拆的好处

- 后面更容易做调试可视化
- 更容易插入日志和诊断信息
- 更容易把“输入问题”和“发布问题”分开定位

---

## 建议修改的设计点

## 7. 新增调试与诊断输出

建议加入可选调试信息：

- 控制器原始位置和旋转增量
- Unity 参考系下的目标线速度和角速度
- 加速度限制前后的速度
- 发布频率
- trigger 激活状态

建议通过布尔开关控制：

- `verboseLog`
- `showDebugGizmos`

这样可以帮助判断“拖手”到底来自：

- 控制器输入
- 死区太大
- scale 太小
- 加速度限制太强
- ROS 端再次滤波

---

## 8. 将左右手逻辑抽象为统一手部状态结构

### 现状

当前左右手变量是平铺写法：

- `leftLastRefPos`
- `rightLastRefPos`
- `leftSmoothVel`
- `rightSmoothVel`

### 建议设计

定义 `HandState` 一类的数据结构，统一保存：

- 是否激活
- 上一帧 pose
- 当前采样 pose
- 目标线速度/角速度
- 当前输出线速度/角速度
- 上次采样时间

### 收益

- 后续维护更简单
- 双手行为更一致
- 更容易加入单手特殊参数

---

## 9. 对触发启停做更平滑的状态切换

### 现状

当前松开 trigger 后会重置缓存并发布零速。

### 建议设计

保留当前行为，但建议明确区分两种模式：

- `release -> immediate zero`
- `release -> short ramp down`

对于机器人遥操作，推荐默认：

- 松开立即归零

如果后续觉得切断太硬，可以加一个很短的减速时间，例如：

- `stopRampTime = 0.05 ~ 0.1s`

---

## 推荐的 Unity 端改造后链路

建议 Unity 内部链路变成：

`Quest2 Controller Pose -> Update 采样 -> 参考系转换 -> 位姿增量 -> 目标 twist -> 死区/缩放 -> 加速度限制 -> ROS 坐标转换 -> FixedUpdate 发布`

这个链路的核心思想是：

- 采样要及时
- 意图要清晰
- 平滑要轻
- 输出要稳

---

## Unity 端推荐参数初值

下面是一组适合作为第一版试跑的 Unity 侧参数，不是最终值，但适合开始调试：

- `linearScale = 0.6`
- `angularScale = 0.8`
- `linearDeadZone = 0.008`
- `angularDeadZone = 0.01`
- `maxLinearIntent = 0.4 ~ 0.8`
- `maxAngularIntent = 0.8 ~ 1.5`
- `maxLinearAccel = 1.5 ~ 3.0`
- `maxAngularAccel = 3.0 ~ 6.0`
- `triggerThreshold = 0.5`

说明：

- 如果感觉“拖手”，优先提高 `maxLinearAccel / maxAngularAccel`
- 如果感觉“机器人太猛”，先减小 `Scale`，其次减小 `Intent Max`
- 如果感觉“静止时抖”，先微调 `DeadZone`

---

## Unity 端不建议承担的职责

下面这些建议留在 ROS 侧：

- 时间戳修正
- 最终 watchdog
- 最终安全限幅
- 工作空间裁剪
- 奇异位形附近降速
- 机器人模型相关保护

原因是这些属于机器人执行安全层，不属于 Quest 2 输入层。

---

## 建议的重构顺序

为了避免一次改太多，建议按下面顺序进行：

1. 保留现有 topic 和消息格式不变
2. 将采样从 `FixedUpdate()` 拆到 `Update()`
3. 引入真实采样时间差
4. 用加速度限制替代当前重 `Lerp`
5. 调整 Inspector 参数命名和职责
6. 增加调试输出
7. 如有需要，再重构为 `HandState`

这样可以一边跑一边验证，不会一下子破坏现有 ROS 对接。

---

## 最终结论

Unity 端最重要的设计修改，不是继续叠加更重的滤波，而是：

- 改善采样时机
- 明确参考坐标系
- 让位姿增量到 twist 的变换更贴近真实输入
- 用加速度限制替代重 `Lerp`
- 把 Unity 参数定义为“意图层参数”

如果这几项做对了，Unity 输出给 ROS 的 `twist_raw` 会更像“稳定、自然、跟手的人手意图”，后面的 ROS bridge 和 MoveIt Servo 也会更容易调。
