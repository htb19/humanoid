from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .callbacks import PeriodicEvalCallback, TrainingMonitorCallback
from .config import load_robot_training_config
from .env import HumanoidBrickPickEnv


def _to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train PPO in Isaac Sim without ROS 2.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--log-root", type=Path, default=repo_root / "logs" / "ppo_runs")
    parser.add_argument("--headless", default="false")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--robot-description-path", type=Path, default=repo_root / "assets" / "robot_description")
    parser.add_argument("--arm-joints", type=str, default=None)
    parser.add_argument("--gripper-joints", type=str, default=None)
    parser.add_argument("--end-effector-link", type=str, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--print-freq-updates", type=int, default=5)
    return parser


def make_env_factory(training_config, headless: bool, evaluation: bool = False):
    def _init():
        env = HumanoidBrickPickEnv(
            training_config=training_config,
            headless=headless,
            evaluation=evaluation,
        )
        return Monitor(
            env,
            info_keywords=(
                "is_success",
                "reached_object",
                "grasped_object",
                "lifted_object",
                "stable_hold",
                "stable_hold_steps",
                "max_brick_height",
                "min_distance_to_brick",
                "episode_reward",
                "episode_length",
            ),
        )

    return _init


def main() -> None:
    args = build_arg_parser().parse_args()
    log_root = args.log_root.resolve()
    robot_description_path = args.robot_description_path.resolve()
    headless = _to_bool(args.headless)
    if args.num_envs < 1:
        raise ValueError("--num-envs must be at least 1")
    if args.num_envs > 1 and not headless:
        raise ValueError("Multi-environment training currently requires --headless true.")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_root / run_name
    checkpoints_dir = run_dir / "checkpoints"
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    training_config = load_robot_training_config(
        repo_root=Path(__file__).resolve().parents[1],
        robot_description_path=robot_description_path,
        arm_joints=args.arm_joints,
        gripper_joints=args.gripper_joints,
        end_effector_link=args.end_effector_link,
    )

    env_fns = [
        make_env_factory(training_config=training_config, headless=headless, evaluation=False)
        for _ in range(args.num_envs)
    ]
    if args.num_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        tensorboard_log=str(run_dir),
        stats_window_size=100,
    )
    logger = configure(str(run_dir), ["stdout", "csv", "tensorboard"])
    model.set_logger(logger)

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=args.checkpoint_freq,
                save_path=str(checkpoints_dir),
                name_prefix="ppo_humanoid_brick",
            ),
            TrainingMonitorCallback(
                run_dir=run_dir,
                print_freq_updates=args.print_freq_updates,
            ),
            PeriodicEvalCallback(
                run_dir=run_dir,
                eval_script=Path(__file__).resolve().parents[1] / "eval_policy.py",
                model_dir=models_dir,
                eval_freq_steps=args.eval_freq,
                n_eval_episodes=args.eval_episodes,
                robot_description_path=robot_description_path,
                headless=headless,
            ),
        ]
    )

    model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
    model.save(str(models_dir / "ppo_humanoid_brick_final"))
    model.save(str(models_dir / "ppo_humanoid_brick_latest"))
    with (models_dir / "recommended_eval_model.txt").open("w", encoding="utf-8") as handle:
        handle.write(str(models_dir / "ppo_humanoid_brick_final.zip"))
    env.close()


if __name__ == "__main__":
    main()
