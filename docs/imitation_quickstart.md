# Imitation Pipeline Quickstart

## 1. Build

```bash
cd ~/humanoid
colcon build --packages-select robot_imitation_pipeline robot_keyboard_control robot_commander
source install/setup.bash
```

## 2. Start The Existing Real-Robot Stack

Use the same launch path already verified for keyboard teleop:

```bash
ros2 launch robot_commander robot_moveit.launch.xml use_simulation:=false
```

If the commander node is not included in your launch session, start it separately:

```bash
ros2 run robot_commander commander
```

## 3. Start Keyboard Teleop

Use joint keyboard control as the v1 expert source:

```bash
ros2 run robot_keyboard_control joint_control
```

## 4. Start The Recorder

In another terminal:

```bash
cd ~/humanoid
source install/setup.bash
ros2 launch robot_imitation_pipeline demo_recorder.launch.py
```

Default save path:

```text
~/humanoid/data/imitation_raw
```

## 5. Record One Episode

Start:

```bash
ros2 run robot_imitation_pipeline demo_control start
```

Operate the robot with keyboard teleop.

Stop and mark success:

```bash
ros2 run robot_imitation_pipeline demo_control stop --success
```

Stop and mark failure:

```bash
ros2 run robot_imitation_pipeline demo_control stop --failure
```

## 6. Validate

```bash
ros2 run robot_imitation_pipeline validate_demo data/imitation_raw
```

Validate one episode:

```bash
ros2 run robot_imitation_pipeline validate_demo data/imitation_raw/episode_000001
```

## 7. Dry-Run Replay

Default replay is non-actuating:

```bash
ros2 run robot_imitation_pipeline replay_demo data/imitation_raw/episode_000001
```

Real robot replay is intentionally gated twice. Copy the replay config, edit the copy so `execute_on_robot: true`, and pass `--execute-on-robot`:

```bash
cp src/robot_imitation_pipeline/config/replay.yaml /tmp/replay_execute.yaml
# Edit /tmp/replay_execute.yaml and set execute_on_robot: true
ros2 run robot_imitation_pipeline replay_demo data/imitation_raw/episode_000001 \
  --config /tmp/replay_execute.yaml \
  --execute-on-robot
```

Before enabling this, inspect the dry-run output and confirm the robot is in a safe state.

## 8. Convert For Training

```bash
ros2 run robot_imitation_pipeline convert_to_hdf5 data/imitation_raw \
  --output-dir data/imitation_converted
```

This writes `dataset.npz`, split metadata, and `dataset.hdf5` if `h5py` is installed.

## 9. Train The First Baseline

```bash
ros2 run robot_imitation_pipeline train_bc \
  --config src/robot_imitation_pipeline/config/training.yaml
```

The first baseline is state-to-action behavior cloning:

- input: 16D robot state vector
- output: 16D keyboard command action vector

Images are recorded now but not used by the first baseline. Add image models only after the raw dataset validates cleanly.

## Useful Topic Checks

```bash
ros2 topic hz /joint_states
ros2 topic hz /head_camera/image
ros2 topic echo /left_joint_command
ros2 topic echo /right_joint_command
```
