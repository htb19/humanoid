from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from .config import load_robot_training_config
from .env import HumanoidBrickPickEnv


def resolve_model_path(model_path: Path) -> Path:
    resolved = model_path.resolve()
    if resolved.is_file():
        return resolved
    candidates = [
        resolved / "best_model" / "best_model.zip",
        resolved / "models" / "ppo_humanoid_brick_final.zip",
        resolved / "models" / "ppo_humanoid_brick_latest.zip",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve a model file from {resolved}")


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Evaluate a PPO policy in Isaac Sim without ROS 2.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--headless", default="false")
    parser.add_argument("--robot-description-path", type=Path, default=repo_root / "assets" / "robot_description")
    parser.add_argument("--arm-joints", type=str, default=None)
    parser.add_argument("--gripper-joints", type=str, default=None)
    parser.add_argument("--end-effector-link", type=str, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rewards = []
    success_count = 0
    reached_count = 0
    grasped_count = 0
    lifted_count = 0
    stable_hold_count = 0

    training_config = load_robot_training_config(
        repo_root=Path(__file__).resolve().parents[1],
        robot_description_path=args.robot_description_path.resolve(),
        arm_joints=args.arm_joints,
        gripper_joints=args.gripper_joints,
        end_effector_link=args.end_effector_link,
    )
    env = HumanoidBrickPickEnv(
        training_config=training_config,
        headless=args.headless,
        evaluation=True,
    )
    model = PPO.load(str(resolve_model_path(args.model)))

    for _ in range(args.episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        while not done and not truncated:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
        rewards.append(float(episode_reward))
        success_count += int(info.get("is_success", False))
        reached_count += int(info.get("reached_object", False))
        grasped_count += int(info.get("grasped_object", False))
        lifted_count += int(info.get("lifted_object", False))
        stable_hold_count += int(info.get("stable_hold", False))

    summary = {
        "episodes": args.episodes,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "success_rate": success_count / args.episodes,
        "reached_rate": reached_count / args.episodes,
        "grasped_rate": grasped_count / args.episodes,
        "lifted_rate": lifted_count / args.episodes,
        "stable_hold_rate": stable_hold_count / args.episodes,
    }
    print(json.dumps(summary, indent=2))
    if args.summary_out is not None:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = args.summary_out.with_suffix(args.summary_out.suffix + ".tmp")
        tmp_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        tmp_path.replace(args.summary_out)
    env.close()


if __name__ == "__main__":
    main()
