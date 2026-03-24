from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.pick_brick_env import PickBrickEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    args = parse_args()
    render_mode = "human" if args.render else None
    env = PickBrickEnv(render_mode=render_mode)
    model = PPO.load(str(args.model_path))

    successes = 0
    returns = []
    max_heights = []

    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        terminated = False
        truncated = False
        episode_return = 0.0
        episode_max_height = info["brick_height"]

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_return += reward
            episode_max_height = max(episode_max_height, info["brick_height"])
            if args.render:
                env.render()

        successes += int(info["success"])
        returns.append(episode_return)
        max_heights.append(episode_max_height)
        print(
            f"Episode {episode + 1}: "
            f"return={episode_return:.3f}, "
            f"success={info['success']}, "
            f"max_brick_height={episode_max_height:.3f}, "
            f"final_distance={info['distance_to_brick']:.3f}"
        )

    print(f"Success rate: {successes / args.episodes:.2%}")
    print(f"Average return: {np.mean(returns):.3f}")
    print(f"Average max brick height: {np.mean(max_heights):.3f}")
    env.close()


if __name__ == "__main__":
    main()
