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


def _parse_float_range(value: str | None, label: str) -> tuple[float, float] | None:
    if value is None:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"--{label} must be formatted as min,max")
    low, high = float(parts[0]), float(parts[1])
    if low > high:
        raise ValueError(f"--{label} lower bound must be <= upper bound")
    return low, high


def _parse_joint_overrides(value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    overrides: dict[str, float] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        name, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError("--reaching-home-overrides must be formatted as joint=value,joint=value")
        overrides[name.strip()] = float(raw_value)
    return overrides


def _build_reaching_brick_range(args: argparse.Namespace) -> dict[str, float] | None:
    x_range = _parse_float_range(args.reaching_brick_x_range, "reaching-brick-x-range")
    y_range = _parse_float_range(args.reaching_brick_y_range, "reaching-brick-y-range")
    if x_range is None and y_range is None:
        return None
    x_min, x_max = x_range if x_range is not None else (0.43, 0.50)
    y_min, y_max = y_range if y_range is not None else (-0.28, -0.20)
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train PPO in Isaac Sim without ROS 2.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--log-root", type=Path, default=repo_root / "logs" / "ppo_runs")
    parser.add_argument("--headless", default="true")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--robot-description-path", type=Path, default=repo_root / "assets" / "robot_description")
    parser.add_argument("--arm-joints", type=str, default=None)
    parser.add_argument("--gripper-joints", type=str, default=None)
    parser.add_argument("--end-effector-link", type=str, default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--print-freq-updates", type=int, default=5)
    parser.add_argument(
        "--reaching-only",
        default="false",
        help="Use an isolated reaching curriculum: success is ee-object distance below threshold.",
    )
    parser.add_argument("--reach-threshold-phase", type=int, default=1, help="Reaching curriculum phase: 1=0.18, 2=0.15, 3=0.12, 4=0.08.")
    parser.add_argument("--reach-threshold", type=float, default=None, help="Override the reached distance threshold in meters.")
    parser.add_argument("--use-grasp-tcp", default=None, help="Measure reaching from the grasp TCP offset instead of the raw wrist link.")
    parser.add_argument("--arm-action-scale", type=float, default=None, help="Override per-step arm joint action scale.")
    parser.add_argument("--reaching-brick-x-range", type=str, default=None, help="Override reaching-only brick x range as min,max.")
    parser.add_argument("--reaching-brick-y-range", type=str, default=None, help="Override reaching-only brick y range as min,max.")
    parser.add_argument("--reaching-home-overrides", type=str, default=None, help="Override reaching-only home joints as joint=value,joint=value.")
    return parser


def make_env_factory(
    training_config,
    headless: bool,
    evaluation: bool = False,
    reaching_only: bool = False,
    reach_distance_threshold: float | None = None,
    reach_threshold_phase: int | None = None,
    use_grasp_tcp: str | bool | None = None,
    arm_action_scale: float | None = None,
    reaching_brick_range: dict[str, float] | None = None,
    reaching_home_overrides: dict[str, float] | None = None,
):
    def _init():
        env = HumanoidBrickPickEnv(
            training_config=training_config,
            headless=headless,
            evaluation=evaluation,
            reaching_only=reaching_only,
            reach_distance_threshold=reach_distance_threshold,
            reach_threshold_phase=reach_threshold_phase,
            use_grasp_tcp=use_grasp_tcp,
            arm_action_scale=arm_action_scale,
            reaching_brick_range=reaching_brick_range,
            reaching_home_overrides=reaching_home_overrides,
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
                "distance_to_brick",
                "brick_height",
                "final_distance_to_brick",
                "final_brick_height",
                "reached_threshold",
                "reach_threshold_phase",
                "min_episode_distance",
                "episode_reward",
                "episode_length",
                "reward_action_penalty",
                "reward_distance",
                "reward_approach",
                "reward_reach_bonus",
                "reward_grasp_bonus",
                "reward_success_bonus",
                "reward_lift_bonus",
                "reward_height_bonus",
                "reward_velocity_penalty",
                "action_magnitude",
                "last_action_magnitude",
                "current_reached_object",
                "current_grasped_object",
                "current_lifted_object",
                "current_stable_hold",
                "current_success",
            ),
        )

    return _init


def main() -> None:
    args = build_arg_parser().parse_args()
    log_root = args.log_root.resolve()
    robot_description_path = args.robot_description_path.resolve()
    headless = _to_bool(args.headless)
    reaching_only = _to_bool(args.reaching_only)
    reaching_brick_range = _build_reaching_brick_range(args)
    reaching_home_overrides = _parse_joint_overrides(args.reaching_home_overrides)
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
        make_env_factory(
            training_config=training_config,
            headless=headless,
            evaluation=False,
            reaching_only=reaching_only,
            reach_distance_threshold=args.reach_threshold,
            reach_threshold_phase=args.reach_threshold_phase,
            use_grasp_tcp=args.use_grasp_tcp,
            arm_action_scale=args.arm_action_scale,
            reaching_brick_range=reaching_brick_range,
            reaching_home_overrides=reaching_home_overrides,
        )
        for _ in range(args.num_envs)
    ]
    if args.num_envs == 1:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    # --- PPO hyperparameters (tuned for manipulation) ---
    # Changes from defaults:
    #   learning_rate: 3e-4 -> 1e-4 (more stable for contact-rich tasks)
    #   n_steps: 2048 -> 4096 (longer rollouts capture full reach-grasp-lift episodes)
    #   batch_size: 256 -> 512 (larger batches reduce gradient noise)
    #   n_epochs: default(10) -> 8 (fewer epochs per update to reduce overfitting)
    #   ent_coef: default(0.0) -> 0.005 (small entropy bonus encourages exploration)
    #   clip_range: default(0.2) -> 0.15 (tighter clipping for stability)
    #   policy_kwargs: 2 hidden layers of 256 (wider than default 64x64)
    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=512,
        n_epochs=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.15,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
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
                reaching_only=reaching_only,
                reach_distance_threshold=args.reach_threshold,
                reach_threshold_phase=args.reach_threshold_phase,
                use_grasp_tcp=args.use_grasp_tcp,
                arm_action_scale=args.arm_action_scale,
                reaching_brick_range=reaching_brick_range,
                reaching_home_overrides=reaching_home_overrides,
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
