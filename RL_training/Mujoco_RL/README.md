# MuJoCo RL Pick-and-Lift Baseline

This project turns the existing `robot_description` into a minimal MuJoCo manipulation baseline for picking and lifting a toy brick from a table with Stable-Baselines3 PPO.

## Robot Description Audit

Files found:

- `robot_description/urdf/humanoid.urdf`
- `robot_description/urdf/humanoid.urdf.xacro`
- `robot_description/urdf/humanoid.ros2_control.xacro`
- `robot_description/urdf/humanoid.gazebo*.xacro`
- 22 STL mesh files under `robot_description/meshes/`

Summary:

- Main URDF: `robot_description/urdf/humanoid.urdf`
- Base link: `base_link`
- Structure: torso base, neck branch, mirrored left and right 6-DoF arms, two prismatic-finger grippers, fixed wrist/head cameras
- End-effector candidate for RL: `right_wrist_yaw_link` with an `ee_site` added near the finger pinch point
- Gripper joints: `right_gripper1_joint` and `right_gripper2_joint` on the active arm
- Mimic joints: `right_gripper2_joint` mimics `right_gripper1_joint` with multiplier `-1.0`; same pattern exists on the left arm
- Mesh paths: all meshes exist, but URDF uses `package://robot_description/...` URIs that are inconvenient for direct MuJoCo loading
- Inertials: present for all links
- Collision geometry: present for all links in the URDF, but mesh collision is heavier than needed for a PPO baseline
- Transmissions: none in `humanoid.urdf`; ROS2 control lives in separate xacro files and is ignored for training
- Joint limits: most revolute joints are `[-3.14, 3.14]`, which is too permissive for a stable first RL baseline

Main compatibility fixes used here:

- Use the right arm only for the first task, while keeping the model minimal and stable
- Rebuild the task robot as a MuJoCo MJCF asset generated from the URDF transforms, inertials, and visuals
- Replace heavy mesh contact with simple primitive collision geoms for the arm and fingers
- Keep two explicit finger joints and drive them symmetrically from one policy gripper command instead of relying on URDF mimic behavior
- Clamp the active joint ranges to more conservative values than the raw URDF

## Project Layout

```text
Mujoco_RL/
  robot_description/
  assets/
    humanoid_right_arm.xml
    pick_brick_scene.xml
  envs/
    __init__.py
    pick_brick_env.py
  train/
    train_ppo.py
  eval/
    play_policy.py
  utils/
    build_mjcf.py
    inspect_robot.py
  scripts/
    setup_conda_env.sh
    verify_install.sh
  README.md
  requirements.txt
```

## Assumptions

- The first manipulation baseline uses only the right arm from the robot description.
- The torso is fixed in the world at `z=0.6` so the natural zero pose starts above the table.
- The table and brick are placed near the right arm zero-pose reach to reduce unnecessary exploration difficulty.
- The observation is low-dimensional state only. No camera observations are used.
- The action is 6 joint-position deltas plus 1 scalar gripper command. This is simpler and more PPO-friendly than torque control for a first working baseline.

## Environment Setup

Use your shell initialization:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
```

Create and install the environment:

```bash
bash ~/Mujoco_RL/scripts/setup_conda_env.sh mujoco_rl
```

Activate later with:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mujoco_rl
```

Quick verification:

```bash
bash ~/Mujoco_RL/scripts/verify_install.sh mujoco_rl
```

Direct CUDA check:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mujoco_rl
python -c "import torch; print(torch.cuda.is_available())"
```

## Build the MuJoCo Assets

The environment auto-generates the MuJoCo XML files on first use, but you can build them explicitly:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mujoco_rl
python ~/Mujoco_RL/utils/build_mjcf.py
```

Inspect the robot summary:

```bash
python ~/Mujoco_RL/utils/inspect_robot.py
```

## Train PPO

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mujoco_rl
python ~/Mujoco_RL/train/train_ppo.py --total-timesteps 500000 --run-name ppo_pick_brick
```

Artifacts are written under `runs/<run-name>/`:

- checkpoints
- best model
- tensorboard logs
- final model

TensorBoard:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mujoco_rl
tensorboard --logdir ~/Mujoco_RL/runs/ppo_pick_brick/tb
```

## Evaluate a Trained Policy

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate mujoco_rl
python ~/Mujoco_RL/eval/play_policy.py --model-path ~/Mujoco_RL/runs/ppo_pick_brick/best_model/best_model.zip --episodes 20 --render
```

## Task Definition

- Scene: anchored robot, table, toy brick, ground plane
- Brick reset: random XY spawn on the table in a small range
- Observation: active joint positions, active joint velocities, gripper opening, end-effector position, brick position, relative vector, brick height
- Action: 6 arm joint delta commands plus 1 gripper command mapped to mirrored finger targets
- Success: brick is grasped and lifted above the success height for several consecutive control steps
- Episode ends on success, timeout, brick drop, or workspace exit

Reward terms in `envs/pick_brick_env.py` are explicit and easy to retune:

- negative end-effector to brick distance
- grasp bonus
- lifting reward
- extra height bonus near success
- final success bonus
- action penalty
- smoothness penalty

## Known Limitations

- The baseline uses a right-arm subset rather than the full humanoid model.
- The finger collision pads are simple box approximations, not exact mesh contact.
- The reward and home pose are tuned for a first working baseline, not optimal final performance.
- There is no domain randomization, curriculum, or camera policy yet.
- Human rendering depends on MuJoCo viewer availability in the active environment.

## Next Steps

- Add a scripted pre-grasp curriculum or demonstrations to speed up learning.
- Tune the home pose and brick spawn range after checking the first rollouts.
- Add a small IK-assisted action wrapper if pure joint-delta exploration is too slow.
- Reintroduce more of the original robot once the right-arm task is stable.
