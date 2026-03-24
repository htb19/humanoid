from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.pick_brick_env import PickBrickEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the MuJoCo pick-brick task.")
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="ppo_pick_brick")
    parser.add_argument("--render-eval", action="store_true")
    return parser.parse_args()


def make_env(seed: int):
    def _factory():
        env = PickBrickEnv(render_mode=None)
        env.reset(seed=seed)
        return Monitor(env)

    return _factory


def main() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    args = parse_args()
    run_dir = PROJECT_ROOT / "runs" / args.run_name
    checkpoints_dir = run_dir / "checkpoints"
    best_model_dir = run_dir / "best_model"
    eval_log_dir = run_dir / "eval_logs"
    tensorboard_dir = run_dir / "tb"
    for directory in [checkpoints_dir, best_model_dir, eval_log_dir, tensorboard_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    env_for_check = PickBrickEnv(render_mode=None)
    check_env(env_for_check, warn=True)
    env_for_check.close()

    train_env = DummyVecEnv([make_env(args.seed)])
    eval_env = DummyVecEnv([make_env(args.seed + 1)])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        device=device,
        seed=args.seed,
        tensorboard_log=str(tensorboard_dir),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256])},
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25_000,
        save_path=str(checkpoints_dir),
        name_prefix="ppo_pick_brick",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(eval_log_dir),
        eval_freq=10_000,
        deterministic=True,
        render=args.render_eval,
    )

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        tb_log_name=args.run_name,
        progress_bar=True,
    )

    final_model_path = run_dir / "final_model.zip"
    model.save(str(final_model_path))
    print(f"Saved final model to {final_model_path}")
    print(f"Training device: {device}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
