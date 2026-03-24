# rl_train

Standalone Isaac Sim + PPO training repository for the humanoid brick-picking task.

## Goals

- No ROS 2 runtime dependency
- No `colcon build`
- No `rclpy`, MoveIt, or ROS pub/sub in the training loop
- Uses only local robot assets copied into this repository

## Layout

- `assets/robot_description/`: URDF and meshes copied from the robot repository
- `train_ppo.py`: training entrypoint
- `eval_policy.py`: evaluation entrypoint
- `rl_train/`: environment, asset parsing, and callbacks

## Requirements

- Isaac Sim 5.x
- Isaac Sim Python with:
  - `stable-baselines3`
  - `gymnasium`
  - `numpy`
  - `pyyaml`
  - `tensorboard`

## Train

```bash
cd /home/arthur/rl_train
~/isaacsim/python.sh /home/arthur/rl_train/train_ppo.py \
  --headless true \
  --timesteps 200000 \
  --num-envs 1 \
  --robot-description-path /home/arthur/rl_train/assets/robot_description
```

Parallel headless training:

```bash
~/isaacsim/python.sh /home/arthur/rl_train/train_ppo.py \
  --headless true \
  --timesteps 200000 \
  --num-envs 4 \
  --robot-description-path /home/arthur/rl_train/assets/robot_description
```

## Evaluate

```bash
~/isaacsim/python.sh /home/arthur/rl_train/eval_policy.py \
  --model /home/arthur/rl_train/logs/ppo_runs/<timestamp>/models/ppo_humanoid_brick_final.zip \
  --headless false \
  --episodes 10 \
  --robot-description-path /home/arthur/rl_train/assets/robot_description
```

## Demo

Visual deterministic pick-and-lift demo:

```bash
~/isaacsim/python.sh /home/arthur/rl_train/demo_pick_brick.py \
  --headless false \
  --run-once true \
  --enable-place false \
  --robot-description-path /home/arthur/rl_train/assets/robot_description
```

You can also pass a run directory instead of a `.zip` file. Evaluation will prefer:

1. `best_model/best_model.zip`
2. `models/ppo_humanoid_brick_final.zip`
3. `models/ppo_humanoid_brick_latest.zip`

## Test URDF Import

Use this before PPO if you want to debug URDF loading in isolation:

```bash
~/isaacsim/python.sh /home/arthur/rl_train/test_import_urdf.py \
  --headless true \
  --robot-description-path /home/arthur/rl_train/assets/robot_description
```

This prints:

- the source URDF path
- the generated runtime URDF path in `/tmp/rl_train/`
- file existence and readability
- the exact import config sent to Isaac
- the resulting prim path if import succeeds

## TensorBoard

```bash
~/isaacsim/python.sh -m tensorboard.main --logdir /home/arthur/rl_train/logs/ppo_runs
```
