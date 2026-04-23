"""Manual dual-arm pose tuning utility for Isaac Sim standalone.

How to run:
  ~/isaacsim/python.sh /home/arthur/Isaac_RL/check_pose.py --headless false

Example commands:
  names
  show
  deg
  rad
  home    # all-zero URDF joint pose
  ready   # configured bent-arm ready pose used by demo/training resets
  quit

Example pose input:
  0 0 0 -1.2 0.6 0  0 0 0 1.2 -0.6 0
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path

import numpy as np

from .brick_pick_demo_support import HumanoidBrickPickDemoScene
from .config import RobotTrainingConfig, load_robot_training_config
from .demo_pick_brick import _to_bool

ARM_JOINT_NAMES = [
    "right_base_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_base_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

SETTLE_STEPS = 30


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Interactive dual-arm pose checker for Isaac Sim standalone.")
    parser.add_argument("--headless", default="false")
    parser.add_argument(
        "--robot-description-path",
        type=Path,
        default=repo_root / "assets" / "robot_description",
    )
    return parser


def load_robot(robot_description_path: Path) -> RobotTrainingConfig:
    arm_joints_arg = ",".join(ARM_JOINT_NAMES)
    return load_robot_training_config(
        repo_root=Path(__file__).resolve().parents[1],
        robot_description_path=robot_description_path.resolve(),
        arm_joints=arm_joints_arg,
    )


def create_scene(training_config: RobotTrainingConfig, headless: bool) -> HumanoidBrickPickDemoScene:
    return HumanoidBrickPickDemoScene(training_config=training_config, headless=headless, setup_cameras=False)


def get_target_joint_indices(scene: HumanoidBrickPickDemoScene) -> list[int]:
    return [scene.dof_names.index(name) for name in ARM_JOINT_NAMES]


def get_joint_limits(training_config: RobotTrainingConfig) -> list[tuple[float, float]]:
    limits = []
    for joint_name in ARM_JOINT_NAMES:
        joint_limit = training_config.joint_limits.get(joint_name, {})
        limits.append(
            (
                float(joint_limit.get("min_position", -3.14)),
                float(joint_limit.get("max_position", 3.14)),
            )
        )
    return limits


def print_joint_state(scene: HumanoidBrickPickDemoScene, target_indices: list[int]) -> None:
    current_positions = np.array(scene.articulation.get_joint_positions(), dtype=np.float64)
    print("[check_pose] current arm joint values")
    for index, joint_name in enumerate(ARM_JOINT_NAMES):
        joint_index = target_indices[index]
        print(f"  {index:02d} {joint_name}: {current_positions[joint_index]: .4f}")


def print_joint_names(scene: HumanoidBrickPickDemoScene, target_indices: list[int]) -> None:
    print("[check_pose] arm joint mapping")
    for index, joint_name in enumerate(ARM_JOINT_NAMES):
        print(f"  {index:02d} -> dof {target_indices[index]:02d} -> {joint_name}")


def print_end_effector_poses(scene: HumanoidBrickPickDemoScene) -> None:
    right_pose = scene.get_link_pose("right_wrist_yaw_link")
    left_pose = scene.get_link_pose("left_wrist_yaw_link")
    print("[check_pose] right ee position", np.round(right_pose.position, 4).tolist())
    print("[check_pose] right ee quaternion_wxyz", np.round(right_pose.quaternion_wxyz, 4).tolist())
    print("[check_pose] left ee position", np.round(left_pose.position, 4).tolist())
    print("[check_pose] left ee quaternion_wxyz", np.round(left_pose.quaternion_wxyz, 4).tolist())


def hold_current_joint_positions(scene: HumanoidBrickPickDemoScene) -> None:
    joint_positions = np.array(scene.articulation.get_joint_positions(), dtype=np.float64)
    scene.articulation.set_joint_velocities(np.zeros_like(joint_positions))
    scene.articulation.apply_action(scene._ArticulationAction(joint_positions=joint_positions))
    print("[check_pose] holding imported joint positions", np.round(joint_positions, 4).tolist())


def apply_pose(
    scene: HumanoidBrickPickDemoScene,
    target_indices: list[int],
    target_values: np.ndarray,
    settle_steps: int = SETTLE_STEPS,
) -> None:
    joint_positions = np.array(scene.articulation.get_joint_positions(), dtype=np.float64)
    for i, joint_index in enumerate(target_indices):
        joint_positions[joint_index] = float(target_values[i])
    scene.articulation.set_joint_positions(joint_positions)
    scene.articulation.set_joint_velocities(np.zeros_like(joint_positions))
    scene.step_world(steps=max(1, settle_steps))
    scene.articulation.set_joint_positions(joint_positions)
    scene.articulation.set_joint_velocities(np.zeros_like(joint_positions))

    print("[check_pose] requested values", np.round(target_values, 4).tolist())
    print_joint_state(scene, target_indices)
    print_end_effector_poses(scene)


def warn_joint_limits(target_values: np.ndarray, limits: list[tuple[float, float]], input_mode: str) -> None:
    violations = []
    for i, value in enumerate(target_values):
        lower, upper = limits[i]
        if value < lower or value > upper:
            violations.append({
                "joint": ARM_JOINT_NAMES[i],
                "value": round(float(value), 4),
                "lower": round(lower, 4),
                "upper": round(upper, 4),
            })
    if violations:
        print(f"[warning] pose exceeds joint limits ({input_mode} input mode)", violations)


def parse_pose_values(raw: str, input_mode: str) -> np.ndarray | None:
    parts = raw.split()
    if len(parts) != len(ARM_JOINT_NAMES):
        print(f"[error] expected {len(ARM_JOINT_NAMES)} values, got {len(parts)}")
        return None
    try:
        values = np.array([float(part) for part in parts], dtype=np.float64)
    except ValueError:
        print("[error] failed to parse numeric joint values")
        return None
    if input_mode == "deg":
        values = np.deg2rad(values)
    return values


def input_worker(command_queue: queue.Queue[str]) -> None:
    while True:
        try:
            command = input("check_pose> ").strip()
        except EOFError:
            command = "quit"
        command_queue.put(command)
        if command in {"quit", "exit"}:
            break


def interactive_loop(scene: HumanoidBrickPickDemoScene, training_config: RobotTrainingConfig) -> None:
    command_queue: queue.Queue[str] = queue.Queue()
    worker = threading.Thread(target=input_worker, args=(command_queue,), daemon=True)
    worker.start()

    target_indices = get_target_joint_indices(scene)
    joint_limits = get_joint_limits(training_config)
    input_mode = "rad"
    home_pose = np.zeros(len(ARM_JOINT_NAMES), dtype=np.float64)
    ready_pose = np.array(
        [training_config.home_joint_positions.get(joint_name, 0.0) for joint_name in ARM_JOINT_NAMES],
        dtype=np.float64,
    )

    print("[check_pose] startup mode: applying initial training pose (ready pose)")
    print("[check_pose] command 'home' applies all-zero joints; command 'ready' applies demo/training ready pose")
    apply_pose(scene, target_indices, ready_pose)
    print_joint_names(scene, target_indices)
    print("[check_pose] input mode: rad")

    running = True
    while running and scene._app.is_running():
        scene.step_world(steps=1)
        try:
            command = command_queue.get_nowait()
        except queue.Empty:
            continue

        if not command:
            continue
        if command in {"quit", "exit"}:
            running = False
            continue
        if command == "show":
            print_joint_state(scene, target_indices)
            print_end_effector_poses(scene)
            continue
        if command == "names":
            print_joint_names(scene, target_indices)
            continue
        if command == "deg":
            input_mode = "deg"
            print("[check_pose] input mode: deg")
            continue
        if command == "rad":
            input_mode = "rad"
            print("[check_pose] input mode: rad")
            continue
        if command == "home":
            apply_pose(scene, target_indices, home_pose)
            continue
        if command == "ready":
            apply_pose(scene, target_indices, ready_pose)
            continue

        values = parse_pose_values(command, input_mode)
        if values is None:
            continue
        warn_joint_limits(values, joint_limits, input_mode)
        apply_pose(scene, target_indices, values)


def run_check_pose(robot_description_path: Path, headless: bool) -> None:
    scene: HumanoidBrickPickDemoScene | None = None
    try:
        training_config = load_robot(robot_description_path)
        scene = create_scene(training_config=training_config, headless=headless)
        hold_current_joint_positions(scene)
        interactive_loop(scene, training_config)
    finally:
        if scene is not None:
            scene.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    run_check_pose(
        robot_description_path=args.robot_description_path.resolve(),
        headless=_to_bool(args.headless),
    )


if __name__ == "__main__":
    main()
