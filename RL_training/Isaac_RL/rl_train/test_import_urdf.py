from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_robot_training_config
from .isaac_import import import_urdf, verify_runtime_urdf


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Standalone Isaac URDF import test.")
    parser.add_argument("--headless", default="true")
    parser.add_argument("--robot-description-path", type=Path, default=repo_root / "assets" / "robot_description")
    parser.add_argument("--arm-joints", type=str, default=None)
    parser.add_argument("--gripper-joints", type=str, default=None)
    parser.add_argument("--end-effector-link", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    headless = str(args.headless).strip().lower() in {"1", "true", "yes", "on"}

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": headless})
    try:
        from omni.isaac.core import World

        config = load_robot_training_config(
            repo_root=Path(__file__).resolve().parents[1],
            robot_description_path=args.robot_description_path.resolve(),
            arm_joints=args.arm_joints,
            gripper_joints=args.gripper_joints,
            end_effector_link=args.end_effector_link,
        )
        print("[urdf] source:", config.urdf_path)
        verify_runtime_urdf(config.runtime_urdf_path)

        world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0)
        world.scene.add_default_ground_plane()
        prim_path = import_urdf(config.runtime_urdf_path)
        world.reset()
        print("[urdf] import succeeded:", prim_path)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
