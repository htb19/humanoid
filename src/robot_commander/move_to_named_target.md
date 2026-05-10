# move_to_named_target

将机器人双臂移动到 SRDF 中预定义的命名姿态（group state）。

## 功能

- 创建 `right_arm` 和 `left_arm` 两个 MoveGroupInterface
- 将双臂同时移动到指定的命名姿态
- 移动完成后节点退出，返回码 0 表示成功，1 表示失败

## 参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `pose_name` | `string` | `"Home"` | 目标姿态名称，需在 SRDF 中定义为 `right_arm` 和 `left_arm` 的 group state |

## 可用姿态

SRDF 中已定义的 `right_arm` / `left_arm` group state：

| 姿态名称 | 说明 |
|----------|------|
| `Home` | 所有关节归零，双臂自然下垂 |
| `Pose_init` | 初始屈臂姿态 |

## 使用示例

### 命令行

```bash
# 移动到 Home（默认）
ros2 run robot_commander move_to_named_target

# 移动到 Pose_init
ros2 run robot_commander move_to_named_target --ros-args -p pose_name:=Pose_init
```

### Launch 文件

```xml
<node pkg="robot_commander" exec="move_to_named_target" name="move_to_named_target">
    <param name="pose_name" value="Pose_init"/>
</node>
```
