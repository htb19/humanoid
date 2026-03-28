from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from .brick_pick_demo_support import HumanoidBrickPickDemoScene
from .config import load_robot_training_config
from .demo_pick_brick import _to_bool, create_scene, load_robot, set_initial_joint_positions


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Open Isaac Sim and inspect the direct initial robot pose.")
    parser.add_argument("--headless", default="false")
    parser.add_argument("--hold-seconds", type=float, default=30.0)
    parser.add_argument(
        "--robot-description-path",
        type=Path,
        default=repo_root / "assets" / "robot_description",
    )
    return parser


def _print_pose_report(scene: HumanoidBrickPickDemoScene) -> None:
    ee_pose = scene.get_end_effector_pose()
    joint_positions = np.array(scene.articulation.get_joint_positions(), dtype=np.float64)
    print("[pose_check] joint index map", {name: idx for idx, name in enumerate(scene.dof_names)})
    print("[pose_check] joint positions", np.round(joint_positions, 4).tolist())
    print("[pose_check] end effector position", np.round(ee_pose.position, 4).tolist())
    print("[pose_check] end effector quaternion_wxyz", np.round(ee_pose.quaternion_wxyz, 4).tolist())
    print("[pose_check] end effector z_axis", np.round(np.array(ee_pose.rotation[:, 2], dtype=np.float64), 4).tolist())


def run_pose_check(robot_description_path: Path, headless: bool, hold_seconds: float) -> None:
    scene: HumanoidBrickPickDemoScene | None = None
    try:
        training_config = load_robot(robot_description_path)
        scene = create_scene(training_config=training_config, headless=headless)
        scene.reset_scene()
        set_initial_joint_positions(scene)
        _print_pose_report(scene)

        end_time = time.time() + max(0.0, hold_seconds)
        while scene._app.is_running() and time.time() < end_time:
            scene.step_world(steps=1)
    finally:
        if scene is not None:
            scene.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    run_pose_check(
        robot_description_path=args.robot_description_path.resolve(),
        headless=_to_bool(args.headless),
        hold_seconds=float(args.hold_seconds),
    )


if __name__ == "__main__":
    main()
