# RL Training 说明

本项目包含两套独立的强化学习训练环境，用于人形机器人砖块抓取任务。两套环境均使用 Stable-Baselines3 PPO 算法，不依赖 ROS 2 运行时。

## 目录结构

```
RL_training/
├── Isaac_RL/       # Isaac Sim 训练环境
│   ├── assets/     # URDF 和 mesh 资源
│   ├── rl_train/   # 环境、配置、回调等核心模块
│   ├── logs/       # 训练日志和模型输出
│   ├── train_ppo.py
│   ├── eval_policy.py
│   ├── demo_pick_brick.py
│   ├── check_initial_pose.py
│   └── check_pose.py
└── Mujoco_RL/      # MuJoCo 训练环境
    ├── assets/     # MJCF 场景文件
    ├── envs/       # Gymnasium 环境
    ├── train/      # 训练脚本
    ├── eval/       # 评估脚本
    └── utils/      # 资源构建和检查工具
```

## Isaac Sim 环境

### 依赖

- Isaac Sim 5.x
- stable-baselines3, gymnasium, numpy, pyyaml, tensorboard

### 训练

```bash
~/isaacsim/python.sh train_ppo.py \
  --headless true \
  --timesteps 200000 \
  --num-envs 4 \
  --robot-description-path assets/robot_description
```

### 评估

```bash
~/isaacsim/python.sh eval_policy.py \
  --model logs/ppo_runs/<timestamp>/models/ppo_humanoid_brick_final.zip \
  --headless false \
  --episodes 10
```

### 演示（确定性抓取）

```bash
~/isaacsim/python.sh demo_pick_brick.py --headless false --run-once true
```

### 姿态检查工具

新增两个姿态调试工具：

- `check_initial_pose.py` — 加载机器人到初始姿态，打印关节位置和末端执行器位姿，用于验证初始配置是否正确。
- `check_pose.py` — 交互式双臂姿态调试工具，支持实时输入关节角度（弧度/角度），查看末端位姿，检测关节限位越界。

```bash
# 检查初始姿态
~/isaacsim/python.sh check_initial_pose.py --headless false --hold-seconds 30

# 交互式姿态调试
~/isaacsim/python.sh check_pose.py --headless false
# 交互命令: names, show, deg, rad, home, quit
# 输入12个关节值（右臂6 + 左臂6）即可驱动机器人
```

### TensorBoard

```bash
~/isaacsim/python.sh -m tensorboard.main --logdir logs/ppo_runs
```

## MuJoCo 环境

### 环境配置

```bash
bash scripts/setup_conda_env.sh mujoco_rl
conda activate mujoco_rl
```

### 训练

```bash
python train/train_ppo.py --total-timesteps 500000 --run-name ppo_pick_brick
```

### 评估

```bash
python eval/play_policy.py \
  --model-path runs/ppo_pick_brick/best_model/best_model.zip \
  --episodes 20 --render
```

### 说明

- 当前仅使用右臂子集进行训练
- 观测空间为低维状态（关节位置、速度、末端位置、砖块位置等）
- 动作空间为6个关节位置增量 + 1个夹爪指令
- 成功条件：砖块被抓起并抬升到指定高度

## 详细文档

- Isaac Sim 详细说明：[RL_training/Isaac_RL/README.md](../RL_training/Isaac_RL/README.md)
- MuJoCo 详细说明：[RL_training/Mujoco_RL/README.md](../RL_training/Mujoco_RL/README.md)
