# Imitation Pipeline Audit

Date: 2026-04-23

## Relevant Packages

- `robot_keyboard_control`: verified keyboard teleoperation entrypoints.
- `robot_commander`: subscribes to high-level command topics and calls MoveIt groups.
- `robot_hardware`: ROS 2 control hardware interface for real motors.
- `robot_description` / `robot_moveit_config`: URDF, ros2_control controllers, MoveIt groups, camera links.
- `camera_calibration`: camera topic defaults and hand-eye calibration tooling.
- `robot_servo_control` / `servo_control`: MoveIt Servo and Unity/VR-oriented servo path. This is not used for the v1 dataset pipeline.
- `robot_imitation_pipeline`: new additive data recording, validation, conversion, and training package.

## Teleop Entrypoints

Keyboard teleop lives in `src/robot_keyboard_control/robot_keyboard_control`.

- `ros2 run robot_keyboard_control joint_control`
  - Primary expert source for v1.
  - Reads the first `/joint_states` message to initialize left arm, right arm, and neck targets.
  - Publishes absolute joint target arrays and gripper open/close booleans.
- `ros2 run robot_keyboard_control cartesian_control`
  - Publishes `robot_interfaces/msg/PoseCommand` relative pose increments.
  - Not used as the default v1 action source because the real verified priority is joint keyboard control.

## Robot Control Entrypoints

`robot_commander/src/commander_node.cpp` subscribes to keyboard command topics and queues MoveIt tasks:

- `open_right_gripper`
- `open_left_gripper`
- `right_joint_command`
- `left_joint_command`
- `right_pose_command`
- `left_pose_command`
- `neck_joint_command`

The commander creates MoveIt groups:

- `right_arm`
- `left_arm`
- `right_gripper`
- `left_gripper`
- `neck`

Launch entrypoint:

```bash
ros2 launch robot_commander robot_moveit.launch.xml use_simulation:=false
```

The launch file starts `ros2_control_node` and spawns:

- `joint_state_broadcaster`
- `right_arm_controller`
- `left_arm_controller`
- `left_gripper_controller`
- `right_gripper_controller`
- `neck_controller`

The commander node is currently commented in the launch file, so it may need to be started separately depending on the operator workflow:

```bash
ros2 run robot_commander commander
```

## Topics And Message Types

Primary command topics from joint keyboard teleop:

| Topic | Type | Meaning |
| --- | --- | --- |
| `/left_joint_command` | `example_interfaces/msg/Float64MultiArray` | 6 absolute left arm joint targets |
| `/right_joint_command` | `example_interfaces/msg/Float64MultiArray` | 6 absolute right arm joint targets |
| `/neck_joint_command` | `example_interfaces/msg/Float64MultiArray` | 2 absolute neck joint targets, order `[pitch, yaw]` |
| `/open_left_gripper` | `example_interfaces/msg/Bool` | `true` open, `false` close |
| `/open_right_gripper` | `example_interfaces/msg/Bool` | `true` open, `false` close |

Cartesian teleop topics:

| Topic | Type | Meaning |
| --- | --- | --- |
| `/left_pose_command` | `robot_interfaces/msg/PoseCommand` | Left end-effector pose command |
| `/right_pose_command` | `robot_interfaces/msg/PoseCommand` | Right end-effector pose command |

Robot state:

| Topic | Type | Meaning |
| --- | --- | --- |
| `/joint_states` | `sensor_msgs/msg/JointState` | State from `joint_state_broadcaster`, backed by `robot_hardware` state interfaces |

Controller command topics produced downstream by MoveIt / ros2_control include joint trajectory controller inputs such as `/left_arm_controller/joint_trajectory`, `/right_arm_controller/joint_trajectory`, and `/neck_controller/joint_trajectory`.

## Joint Names

V1 recorder order:

1. `left_base_pitch_joint`
2. `left_shoulder_roll_joint`
3. `left_shoulder_yaw_joint`
4. `left_elbow_pitch_joint`
5. `left_wrist_pitch_joint`
6. `left_wrist_yaw_joint`
7. `right_base_pitch_joint`
8. `right_shoulder_roll_joint`
9. `right_shoulder_yaw_joint`
10. `right_elbow_pitch_joint`
11. `right_wrist_pitch_joint`
12. `right_wrist_yaw_joint`
13. `neck_pitch_joint`
14. `neck_yaw_joint`
15. `left_gripper1_joint`
16. `right_gripper1_joint`

## Camera Streams

Camera topic defaults were found in `camera_calibration` configs and simulation bridge launch files:

- `/head_camera/image`
- `/head_camera/camera_info`
- `/left_camera/image`
- `/left_camera/camera_info`
- `/right_camera/image`
- `/right_camera/camera_info`

The recorder enables only `/head_camera/image` by default. Wrist cameras are configured but disabled until the real camera setup is verified.

## Action Space

The v1 action is captured at the high-level keyboard command interface, not at the lower-level controller trajectory interface.

Chosen action mode: `joint_position_targets`

Action vector shape: `(16,)`

- `[0:6]`: latest left arm joint target from `/left_joint_command`
- `[6:12]`: latest right arm joint target from `/right_joint_command`
- `[12:14]`: latest neck joint target from `/neck_joint_command`
- `[14:16]`: latest gripper open command from `/open_left_gripper` and `/open_right_gripper`

This preserves the verified keyboard control semantics. It does not smooth, densify, or reinterpret sparse keyboard commands.

## Reset / Initialization Logic

- `joint_keyboard_control.py` waits up to 5 seconds for `/joint_states` and initializes its internal arm and neck targets from the real robot state.
- `robot_hardware` reads initial motor positions on activation and initializes command buffers to the current measured position.
- `robot_commander/src/move_to_pose_init_node.cpp` provides named-target initialization logic for MoveIt groups, but this is separate from the keyboard teleop loop.

## Existing Logging / Rosbag Usage

No active rosbag recording path was found in the current source tree. The new recorder bypasses rosbag and writes a simple episode folder format directly for imitation learning. Rosbag can still be used independently for debugging, but the training converter expects the new episode folders.

## Notes

- VR/Unity servo topics exist under `robot_servo_control`, including `/unity/left_twist_raw`, `/unity/right_twist_raw`, and MoveIt Servo twist topics. These are intentionally not used for v1.
- The new pipeline is additive and does not modify the working keyboard teleop or robot commander path.
