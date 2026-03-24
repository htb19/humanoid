from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

#if __package__ in (None, ""):
    # Support execution as a plain script via `~/isaacsim/python.sh train_ppo.py`.
    # In that mode Python does not know about the `robot_rl_training` package unless
    # we add the source roots explicitly.
#    this_file = Path(__file__).resolve()
#    package_root = this_file.parent.parent
#    workspace_root = package_root.parent.parent
#    sys.path.insert(0, str(package_root.parent))
#    sys.path.insert(0, str(workspace_root / "src"))
#    from robot_rl_training.config import load_robot_training_config
#    from robot_rl_training.callbacks import PeriodicEvalCallback, TrainingMonitorCallback
#    from robot_rl_training.env import HumanoidBrickPickEnv
#else:
#from .config import load_robot_training_config
#from .callbacks import PeriodicEvalCallback, TrainingMonitorCallback
#from .env import HumanoidBrickPickEnv

from robot_rl_training.config import load_robot_training_config
from robot_rl_training.callbacks import PeriodicEvalCallback, TrainingMonitorCallback
from robot_rl_training.env import HumanoidBrickPickEnv


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train PPO for Isaac-native brick picking.")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--log-root", type=Path, default=Path("/home/arthur/humanoid/logs/ppo_runs"))
    parser.add_argument("--launch-isaac", default="false")
    parser.add_argument("--headless", default="false")
    parser.add_argument("--workspace-root", type=Path, default=Path("/home/arthur/humanoid"))
    parser.add_argument("--robot-description-path", type=Path, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--print-freq-updates", type=int, default=5)
    return parser


def make_env(
    training_config,
    headless: bool,
    evaluation: bool = False,
):
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


def main() -> None:
    args = build_arg_parser().parse_args()
    args.workspace_root = args.workspace_root.resolve()
    log_root = args.log_root.resolve()
    robot_description_path = args.robot_description_path.resolve() if args.robot_description_path else None
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = log_root / run_name
    checkpoints_dir = run_dir / "checkpoints"
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    headless = _bool_arg(args.headless)
    if _bool_arg(args.launch_isaac):
        print("[train] --launch-isaac is ignored because this script is already the Isaac Sim process.")

    training_config = load_robot_training_config(
        workspace_root=args.workspace_root,
        robot_description_path=robot_description_path,
    )
    try:
        env = DummyVecEnv([lambda: make_env(training_config=training_config, headless=headless, evaluation=False)])
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

        checkpoint = CheckpointCallback(
            save_freq=args.checkpoint_freq,
            save_path=str(checkpoints_dir),
            name_prefix="ppo_humanoid_brick",
        )
        train_monitor = TrainingMonitorCallback(
            run_dir=run_dir,
            print_freq_updates=args.print_freq_updates,
        )
        eval_callback = PeriodicEvalCallback(
            run_dir=run_dir,
            eval_script=Path(__file__).resolve().parent / "eval_policy.py",
            model_dir=models_dir,
            eval_freq_steps=args.eval_freq,
            n_eval_episodes=args.eval_episodes,
            workspace_root=args.workspace_root,
            robot_description_path=robot_description_path,
            headless=headless,
        )

        callbacks = CallbackList([checkpoint, train_monitor, eval_callback])

        model.learn(total_timesteps=args.timesteps, callback=callbacks, progress_bar=True)
        model.save(str(models_dir / "ppo_humanoid_brick_final"))
        model.save(str(models_dir / "ppo_humanoid_brick_latest"))
        env.close()
    finally:
        pass


if __name__ == "__main__":
    main()
