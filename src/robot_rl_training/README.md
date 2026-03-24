# robot_rl_training

Isaac-native PPO training package for the humanoid robot to pick up a toy brick while reusing robot assets from the ROS repository.

## What it uses

- `robot_description`: loads the humanoid URDF and meshes
- `robot_moveit_config`: loads the SRDF group definitions and joint limits as static config
- Isaac Sim: imports the robot URDF, creates the table + brick scene, and steps the articulation locally
- Stable-Baselines3 PPO: trains a policy over right-arm and right-gripper actions

## Dependencies

- Isaac Sim 5.x with `~/isaacsim/python.sh`
- ROS 2 workspace built with `colcon build`
- Python packages in the Isaac/ROS environment:
  - `stable-baselines3`
  - `gymnasium`
  - `numpy`
  - `pyyaml`

## Build

```bash
cd /home/arthur/humanoid
colcon build --packages-select robot_rl_training
source /home/arthur/humanoid/install/setup.bash
```

## Train

```bash
ros2 launch robot_rl_training isaac_rl_training.launch.py \
  headless:=true \
  timesteps:=200000 \
  start_moveit:=false
```

Or run directly:

```bash
~/isaacsim/python.sh /home/arthur/humanoid/src/robot_rl_training/robot_rl_training/train_ppo.py \
  --headless true \
  --timesteps 200000 \
  --workspace-root /home/arthur/humanoid \
  --robot-description-path /home/arthur/humanoid/src/robot_description
```

## Evaluate

```bash
~/isaacsim/python.sh /home/arthur/humanoid/src/robot_rl_training/robot_rl_training/eval_policy.py \
  --model /home/arthur/humanoid/logs/ppo_runs/<timestamp>/models/ppo_humanoid_brick_final.zip \
  --headless true \
  --episodes 10 \
  --workspace-root /home/arthur/humanoid \
  --robot-description-path /home/arthur/humanoid/src/robot_description
```

## Monitoring

- Training runs are stored under `logs/ppo_runs/<timestamp>/`
- TensorBoard scalars are written there directly by SB3
- PPO internals such as KL, entropy, policy loss, value loss, clip fraction, learning rate, and explained variance are logged through the SB3 logger
- Task metrics such as success rate, reached rate, grasped rate, lifted rate, and stable hold rate are logged through a custom callback
- Per-update summaries are saved to `update_metrics.csv`
- Periodic evaluation summaries are saved to `evaluations/*.json`
- Periodic checkpoints are saved to `checkpoints/`
- The best model by evaluation success rate is saved to `best_model/best_model.zip`

Launch TensorBoard:

```bash
~/isaacsim/python.sh -m tensorboard.main --logdir /home/arthur/humanoid/logs/ppo_runs
```

## Notes

- The default PPO path does not import `rclpy` and does not use ROS 2 topics or MoveIt in the training loop.
- The active manipulation groups are `right_arm` and `right_gripper`, taken from `robot_moveit_config/config/humanoid.srdf`.
- Joint limits come from `robot_moveit_config/config/joint_limits.yaml`.
- Mesh paths are rewritten from `package://robot_description/...` into absolute filesystem paths for the Isaac runtime copy of the URDF.
- `ros2_interface.py` and `isaac_scene.py` remain optional legacy bridge code and are not used by the default trainer.
