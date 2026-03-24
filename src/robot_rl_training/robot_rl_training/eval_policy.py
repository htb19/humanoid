from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from stable_baselines3 import PPO

if __package__ in (None, ""):
    this_file = Path(__file__).resolve()
    package_root = this_file.parent.parent
    workspace_root = package_root.parent.parent
    sys.path.insert(0, str(package_root.parent))
    sys.path.insert(0, str(workspace_root / "src"))
    from robot_rl_training.config import load_robot_training_config
    from robot_rl_training.env import HumanoidBrickPickEnv
else:
    from .config import load_robot_training_config
    from .env import HumanoidBrickPickEnv


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO brick-pick policy in Isaac Sim.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--headless", default="false")
    parser.add_argument("--workspace-root", type=Path, default=Path("/home/arthur/humanoid"))
    parser.add_argument("--robot-description-path", type=Path, default=None)
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
        workspace_root=args.workspace_root.resolve(),
        robot_description_path=args.robot_description_path.resolve() if args.robot_description_path else None,
    )
    env = HumanoidBrickPickEnv(
        evaluation=True,
        headless=args.headless,
        training_config=training_config,
    )
    model = PPO.load(str(args.model))

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
        if info.get("is_success", False):
            success_count += 1
        if info.get("reached_object", False):
            reached_count += 1
        if info.get("grasped_object", False):
            grasped_count += 1
        if info.get("lifted_object", False):
            lifted_count += 1
        if info.get("stable_hold", False):
            stable_hold_count += 1

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
        args.summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
