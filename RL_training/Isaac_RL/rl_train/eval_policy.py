from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

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
    x_min, x_max = x_range if x_range is not None else (0.41, 0.52)
    y_min, y_max = y_range if y_range is not None else (-0.30, -0.18)
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}


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
    parser.add_argument(
        "--reaching-only",
        default="false",
        help="Evaluate with reaching-only task logic.",
    )
    parser.add_argument("--reach-threshold-phase", type=int, default=1, help="Reaching curriculum phase: 1=0.18, 2=0.15, 3=0.12, 4=0.08.")
    parser.add_argument("--reach-threshold", type=float, default=None, help="Override the reached distance threshold in meters.")
    parser.add_argument("--use-grasp-tcp", default=None, help="Measure reaching from the grasp TCP offset instead of the raw wrist link.")
    parser.add_argument("--arm-action-scale", type=float, default=None, help="Override per-step arm joint action scale.")
    parser.add_argument("--reaching-brick-x-range", type=str, default=None, help="Override reaching-only brick x range as min,max.")
    parser.add_argument("--reaching-brick-y-range", type=str, default=None, help="Override reaching-only brick y range as min,max.")
    parser.add_argument("--reaching-home-overrides", type=str, default=None, help="Override reaching-only home joints as joint=value,joint=value.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rewards = []
    final_distances = []
    final_heights = []
    action_magnitudes = []
    success_count = 0
    reached_count = 0
    grasped_count = 0
    lifted_count = 0
    stable_hold_count = 0
    min_distances = []
    final_info = {}
    reaching_brick_range = _build_reaching_brick_range(args)
    reaching_home_overrides = _parse_joint_overrides(args.reaching_home_overrides)

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
        reaching_only=_to_bool(args.reaching_only),
        reach_distance_threshold=args.reach_threshold,
        reach_threshold_phase=args.reach_threshold_phase,
        use_grasp_tcp=args.use_grasp_tcp,
        arm_action_scale=args.arm_action_scale,
        reaching_brick_range=reaching_brick_range,
        reaching_home_overrides=reaching_home_overrides,
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
        final_distances.append(float(info.get("final_distance_to_brick", info.get("distance_to_brick", 0.0))))
        min_distances.append(float(info.get("min_episode_distance", 0.0)))
        final_heights.append(float(info.get("final_brick_height", info.get("brick_height", 0.0))))
        action_magnitudes.append(float(info.get("action_magnitude", 0.0)))
        final_info = info

    summary = {
        "episodes": args.episodes,
        "reached_threshold": float(final_info.get("reached_threshold", env.reach_distance_threshold)),
        "reach_threshold_phase": int(final_info.get("reach_threshold_phase", env.reach_threshold_phase or 0)),
        "end_effector_link": env.training_config.end_effector_link,
        "end_effector_reference": env.end_effector_reference_name,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "success_rate": success_count / args.episodes,
        "reached_rate": reached_count / args.episodes,
        "grasped_rate": grasped_count / args.episodes,
        "lifted_rate": lifted_count / args.episodes,
        "stable_hold_rate": stable_hold_count / args.episodes,
        "mean_final_distance": sum(final_distances) / len(final_distances) if final_distances else 0.0,
        "mean_min_episode_distance": sum(min_distances) / len(min_distances) if min_distances else 0.0,
        "mean_final_brick_height": sum(final_heights) / len(final_heights) if final_heights else 0.0,
        "mean_action_magnitude": sum(action_magnitudes) / len(action_magnitudes) if action_magnitudes else 0.0,
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
