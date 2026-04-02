# Humanoid Project

This repository contains the codebase for a humanoid robotics project.

It includes the core system implementation as well as supporting resources developed during the project, such as documentation, tools, and testing code.

## Repository Structure

- `src/`  
  Core ROS 2 source code and packages intended for direct build and execution (robot_description, robot_commander, moveit config, hardware interface, simulation, etc.).

- `RL_training/`  
  Reinforcement learning training code for the humanoid brick-picking task. Contains two independent sub-projects:
  - `Isaac_RL/` — Isaac Sim + Stable-Baselines3 PPO training pipeline (standalone, no ROS dependency).
  - `Mujoco_RL/` — MuJoCo + Stable-Baselines3 PPO baseline using a right-arm subset.

- `docs/`  
  Project documentation, including environment setup, control instructions, calibration guides, and RL training notes.

- `resource/`  
  Auxiliary tools and experimental code used during development (e.g. motor driver testing).

## Quick Links

| Topic | Doc |
|---|---|
| Simulation environment setup | [docs/仿真环境配置.md](docs/仿真环境配置.md) |
| Robot commander & MoveIt | [docs/robot_commander_README.md](docs/robot_commander_README.md) |
| Keyboard teleoperation | [docs/keyboard_control_README.md](docs/keyboard_control_README.md) |
| Servo control | [docs/伺服控制说明.md](docs/伺服控制说明.md) |
| Hand-eye calibration | [docs/手眼标定说明.md](docs/手眼标定说明.md) |
| RL training overview | [docs/RL_training.md](docs/RL_training.md) |
| Known issues | [docs/Problem.md](docs/Problem.md) |

## Notes

Some contents in this repository are intended for development and testing purposes and are not part of the final runtime system.

## Contributors

Zhangyu Fan: Hardware

Tongbin Hu: Software

Zhe Wang: Software

Wu Jie: Software
