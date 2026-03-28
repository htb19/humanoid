"""Standalone Isaac Sim brick-pick demo.

How to run:
  GUI:
    ~/isaacsim/python.sh /home/arthur/Isaac_RL/demo_pick_brick.py --headless false --run-once true --enable-place false
  Headless:
    ~/isaacsim/python.sh /home/arthur/Isaac_RL/demo_pick_brick.py --headless true --run-once true --enable-place false

Assumptions:
  - The robot description package exists at assets/robot_description unless overridden.
  - The current workspace already contains a URDF import path that Isaac Sim can load.
  - Gripper attachment logic is approximate; if gripper joints are missing, the demo falls back to safe arm motions only.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import time

import numpy as np

from .brick_pick_demo_support import HumanoidBrickPickDemoScene, parse_joint_origins
from .config import RobotTrainingConfig, load_robot_training_config
from .pose_math import Pose, compose_pose, invert_pose, rotation_matrix_from_axes


# Motion speed tuning for the standalone Isaac Sim demo.
# Lower MOTION_SPEED_SCALE makes each update move a smaller fraction toward the target.
# STEPS_PER_TARGET and SETTLE_STEPS make the motion easier to tune later without refactoring.
MOTION_SPEED_SCALE = 0.05
STEPS_PER_TARGET = 12
SETTLE_STEPS = 80
MAX_DEMO_SECONDS = 45.0

_READY_ARM_POSE = {
    # Validated dual-arm working pose from simulation.
    # Joint order:
    # [right_base_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow_pitch,
    #  right_wrist_pitch, right_wrist_yaw, left_base_pitch, left_shoulder_roll,
    #  left_shoulder_yaw, left_elbow_pitch, left_wrist_pitch, left_wrist_yaw]
    "right_base_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_pitch_joint": -1.2,
    "right_wrist_pitch_joint": 0.6,
    "right_wrist_yaw_joint": 0.0,
    "left_base_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_pitch_joint": 1.2,
    "left_wrist_pitch_joint": -0.6,
    "left_wrist_yaw_joint": 0.0,
}

_JOINT_INDEX_DEBUG_PRINTED = False


def _to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _home_arm_configuration(arm_joint_names: list[str]) -> np.ndarray:
    return np.array([_READY_ARM_POSE.get(joint_name, 0.0) for joint_name in arm_joint_names], dtype=np.float64)


def build_ready_pose(scene: HumanoidBrickPickDemoScene, current_joint_positions: np.ndarray) -> np.ndarray:
    joint_positions = np.array(current_joint_positions, dtype=np.float64)
    joint_name_to_index = {name: index for index, name in enumerate(scene.dof_names)}

    table_center = np.array(
        [
            scene.scene_config.table_position[0],
            scene.scene_config.table_position[1],
            scene.scene_config.table_height,
        ],
        dtype=np.float64,
    )
    base_pose = scene.get_base_pose()
    _log(
        "ready_pose",
        "Building table-facing ready pose.",
        robot_base_position=np.round(base_pose.position, 4).tolist(),
        table_position=np.round(table_center, 4).tolist(),
    )

    for joint_name, value in _READY_ARM_POSE.items():
        joint_index = joint_name_to_index.get(joint_name)
        if joint_index is None:
            continue
        joint_positions[joint_index] = value
    return joint_positions


def _check_arm_pose_symmetry(joint_name_to_index: dict[str, int], joint_positions: np.ndarray) -> None:
    mirrored_pairs = [
        ("base_pitch", "same"),
        ("shoulder_roll", "same"),
        ("shoulder_yaw", "same"),
        ("elbow_pitch", "opposite"),
        ("wrist_pitch", "opposite"),
        ("wrist_yaw", "same"),
    ]
    mismatches = {}
    for suffix, mode in mirrored_pairs:
        right_name = f"right_{suffix}_joint"
        left_name = f"left_{suffix}_joint"
        if right_name not in joint_name_to_index or left_name not in joint_name_to_index:
            continue
        right_value = float(joint_positions[joint_name_to_index[right_name]])
        left_value = float(joint_positions[joint_name_to_index[left_name]])
        metric = abs(right_value - left_value) if mode == "same" else abs(right_value + left_value)
        if metric > 0.15:
            mismatches[suffix] = {
                "right": round(right_value, 4),
                "left": round(left_value, 4),
                "mode": mode,
                "error": round(metric, 4),
            }
    if mismatches:
        print("[warning] initial arm symmetry mismatch", mismatches)


def _log(stage: str, message: str, **details: object) -> None:
    payload = {"stage": stage, "message": message}
    if details:
        payload["details"] = details
    print("[demo]", payload)


class TrajectoryRecorder:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.records: list[dict[str, object]] = []
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._debug_draw = None
        self._draw_enabled = False
        self._draw_stride = 5
        self._record_count = 0

        try:
            from omni.isaac.debug_draw import _debug_draw

            self._debug_draw = _debug_draw.acquire_debug_draw_interface()
            self._draw_enabled = self._debug_draw is not None
        except Exception:
            self._debug_draw = None
            self._draw_enabled = False

    def record(
        self,
        timestamp: float,
        ee_pose: Pose,
        joint_positions: np.ndarray,
        target_pose: Pose | None = None,
    ) -> None:
        self._record_count += 1
        record = {
            "time": float(timestamp),
            "position": np.array(ee_pose.position, dtype=np.float64),
            "quaternion_wxyz": np.array(ee_pose.quaternion_wxyz, dtype=np.float64),
            "joint_positions": np.array(joint_positions, dtype=np.float64),
            "target_position": None if target_pose is None else np.array(target_pose.position, dtype=np.float64),
        }
        self.records.append(record)

        if self._draw_enabled and self._record_count % self._draw_stride == 0:
            try:
                self._debug_draw.draw_points(
                    [tuple(float(v) for v in ee_pose.position)],
                    [(0.1, 0.8, 1.0, 1.0)],
                    [6.0],
                )
                if target_pose is not None:
                    self._debug_draw.draw_points(
                        [tuple(float(v) for v in target_pose.position)],
                        [(1.0, 0.85, 0.1, 1.0)],
                        [8.0],
                    )
            except Exception:
                self._draw_enabled = False

    def save(self) -> None:
        if not self.records:
            return

        npy_path = self.log_dir / 'ee_trajectory.npy'
        csv_path = self.log_dir / 'ee_trajectory.csv'

        np.save(npy_path, np.array(self.records, dtype=object), allow_pickle=True)

        joint_count = len(self.records[0]["joint_positions"])
        header = ["time", "x", "y", "z", "qw", "qx", "qy", "qz", *[f"joint{i + 1}" for i in range(joint_count)]]
        with csv_path.open('w', encoding='utf-8', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(header)
            for record in self.records:
                position = record["position"]
                quaternion = record["quaternion_wxyz"]
                joints = record["joint_positions"]
                writer.writerow([
                    record["time"],
                    position[0],
                    position[1],
                    position[2],
                    quaternion[0],
                    quaternion[1],
                    quaternion[2],
                    quaternion[3],
                    *joints.tolist(),
                ])

        _log('trajectory', 'Saved end-effector trajectory.', npy_path=str(npy_path), csv_path=str(csv_path), samples=len(self.records))


class DemoState(Enum):
    RESET = auto()
    MOVE_TO_PREGRASP = auto()
    MOVE_TO_GRASP = auto()
    CLOSE_GRIPPER = auto()
    WAIT_FOR_ATTACH = auto()
    LIFT = auto()
    MOVE_TO_PLACE_PREGRASP = auto()
    MOVE_TO_PLACE = auto()
    OPEN_GRIPPER = auto()
    RETRACT = auto()
    DONE = auto()


@dataclass(frozen=True)
class GeometryPlan:
    brick_pose: Pose
    grasp_pose: Pose
    pregrasp_pose: Pose
    lift_pose: Pose
    place_pose: Pose | None
    place_pregrasp_pose: Pose | None
    joint_targets: dict[DemoState, np.ndarray]


@dataclass(frozen=True)
class WorkspaceLimits:
    shoulder_position: np.ndarray
    radial_min: float
    radial_max: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


@dataclass
class DemoTuning:
    arm_tolerance: float = 0.04
    gripper_tolerance: float = 0.005
    close_gripper_seconds: float = 2.5
    attach_wait_seconds: float = 1.5
    done_hold_seconds: float = 3.0
    step_dt: float = 1.0 / 60.0
    pregrasp_height: float = 0.10
    lift_height: float = 0.18
    place_height: float = 0.10
    attach_threshold_xy: float = 0.05
    attach_threshold_z: float = 0.06
    wrist_to_grasp_translation: np.ndarray | None = None
    wrist_to_grasp_rotation: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.wrist_to_grasp_translation is None:
            self.wrist_to_grasp_translation = np.array([0.0, 0.141, -0.0425], dtype=np.float64)
        if self.wrist_to_grasp_rotation is None:
            self.wrist_to_grasp_rotation = np.eye(3, dtype=np.float64)


def load_robot(robot_description_path: Path) -> RobotTrainingConfig:
    resolved_path = robot_description_path.resolve()
    _log("load_robot", "Resolving robot description path.", robot_description_path=str(resolved_path))
    if not resolved_path.exists():
        raise FileNotFoundError(f"Robot description path does not exist: {resolved_path}")
    if not resolved_path.is_dir():
        raise NotADirectoryError(f"Robot description path is not a directory: {resolved_path}")
    if not (resolved_path / "urdf").exists():
        raise FileNotFoundError(f"Robot description path is missing the urdf directory: {resolved_path / 'urdf'}")

    training_config = load_robot_training_config(
        repo_root=Path(__file__).resolve().parents[1],
        robot_description_path=resolved_path,
    )
    _log(
        "load_robot",
        "Resolved robot assets.",
        urdf_path=str(training_config.urdf_path),
        runtime_urdf_path=str(training_config.runtime_urdf_path),
        end_effector_link=training_config.end_effector_link,
    )
    return training_config


def create_scene(training_config: RobotTrainingConfig, headless: bool) -> HumanoidBrickPickDemoScene:
    _log("create_scene", "Creating Isaac Sim world.", headless=headless)
    scene = HumanoidBrickPickDemoScene(training_config=training_config, headless=headless)
    _log("create_scene", "Scene created.", robot_prim_path=scene.robot_prim_path)
    return scene


def spawn_brick(scene: HumanoidBrickPickDemoScene) -> None:
    brick_pose = scene.get_brick_pose()
    _log("spawn_brick", "Brick ready in scene.", brick_position=np.round(brick_pose.position, 4).tolist())


def set_initial_joint_positions(scene: HumanoidBrickPickDemoScene) -> None:
    global _JOINT_INDEX_DEBUG_PRINTED

    # Set the robot articulation directly to the ready pose during initialization/reset.
    # This avoids moving to the pose through a controller sequence.
    current_joint_positions = np.array(scene.articulation.get_joint_positions(), dtype=np.float64)
    joint_name_to_index = {name: index for index, name in enumerate(scene.dof_names)}

    if not _JOINT_INDEX_DEBUG_PRINTED:
        _log(
            "set_initial_joint_positions",
            "Articulation joint index map.",
            joint_indices={name: index for index, name in enumerate(scene.dof_names)},
        )
        _JOINT_INDEX_DEBUG_PRINTED = True

    joint_positions = build_ready_pose(scene, current_joint_positions)
    applied = {
        joint_name: _READY_ARM_POSE[joint_name]
        for joint_name in _READY_ARM_POSE
        if joint_name in joint_name_to_index
    }

    _log(
        "set_initial_joint_positions",
        "Setting articulation joints directly for the initial ready pose.",
        applied_joints=applied,
    )
    if hasattr(scene.articulation, "set_joints_default_state"):
        scene.articulation.set_joints_default_state(positions=joint_positions)
        scene.world.reset()
        scene.articulation.initialize()
    scene.articulation.set_joint_positions(joint_positions)
    scene.articulation.set_joint_velocities(np.zeros_like(joint_positions))
    scene.step_world(steps=SETTLE_STEPS)
    scene.articulation.set_joint_positions(joint_positions)
    scene.articulation.set_joint_velocities(np.zeros_like(joint_positions))

    current_positions = np.array(scene.articulation.get_joint_positions(), dtype=np.float64)
    base_pose = scene.get_base_pose()
    table_center = np.array(
        [
            scene.scene_config.table_position[0],
            scene.scene_config.table_position[1],
            scene.scene_config.table_height,
        ],
        dtype=np.float64,
    )
    right_ee_pose = scene.get_link_pose("right_wrist_yaw_link")
    left_ee_pose = scene.get_link_pose("left_wrist_yaw_link")
    right_distance = float(np.linalg.norm(right_ee_pose.position - table_center))
    left_distance = float(np.linalg.norm(left_ee_pose.position - table_center))

    _log(
        "set_initial_joint_positions",
        "Current articulation state after direct pose set.",
        robot_base_position=np.round(base_pose.position, 4).tolist(),
        table_position=np.round(table_center, 4).tolist(),
        joint_positions=np.round(current_positions, 4).tolist(),
        right_ee_position=np.round(right_ee_pose.position, 4).tolist(),
        right_ee_quaternion=np.round(right_ee_pose.quaternion_wxyz, 4).tolist(),
        right_ee_z_axis=np.round(np.array(right_ee_pose.rotation[:, 2], dtype=np.float64), 4).tolist(),
        left_ee_position=np.round(left_ee_pose.position, 4).tolist(),
        left_ee_quaternion=np.round(left_ee_pose.quaternion_wxyz, 4).tolist(),
        left_ee_z_axis=np.round(np.array(left_ee_pose.rotation[:, 2], dtype=np.float64), 4).tolist(),
        right_distance_to_table=round(right_distance, 4),
        left_distance_to_table=round(left_distance, 4),
    )
    if float(right_ee_pose.rotation[2, 2]) > 0.0:
        print("[warning] right EE is not facing downward toward the table")
    if float(left_ee_pose.rotation[2, 2]) > 0.0:
        print("[warning] left EE is not facing downward toward the table")
    if right_distance > 0.65:
        print("[warning] right arm is too far from the tabletop workspace")
    if left_distance > 0.75:
        print("[warning] left arm is too far from the tabletop workspace")
    _check_arm_pose_symmetry(joint_name_to_index, current_positions)


class BrickPickDemo:
    def __init__(
        self,
        training_config: RobotTrainingConfig,
        scene: HumanoidBrickPickDemoScene,
        run_once: bool,
        enable_place: bool,
    ) -> None:
        self.training_config = training_config
        self.scene = scene
        self.run_once = run_once
        self.enable_place = enable_place
        self.tuning = DemoTuning()

        self.arm_dim = len(self.training_config.arm_joints)
        self.gripper_dim = len(self.training_config.gripper_joints)
        self.open_gripper = np.array([0.03, -0.03], dtype=np.float64)[: self.gripper_dim]
        self.closed_gripper = np.array([0.0, 0.0], dtype=np.float64)[: self.gripper_dim]
        self.motion_only_mode = self.gripper_dim == 0

        self.state = DemoState.RESET
        self.state_start_time = time.time()
        self.demo_start_time = time.time()
        self.demo_count = 0
        self.attached = False
        self.plan: GeometryPlan | None = None
        self._ik_targets: dict[DemoState, np.ndarray] = {}
        self._last_debug_time = 0.0
        self.trajectory_recorder = TrajectoryRecorder(Path(__file__).resolve().parents[1] / "logs")
        self._trajectory_saved = False

        self.wrist_to_grasp = Pose(
            position=self.tuning.wrist_to_grasp_translation,
            rotation=self.tuning.wrist_to_grasp_rotation,
        )
        self.grasp_to_wrist = invert_pose(self.wrist_to_grasp)
        self.workspace = self._derive_workspace_limits()
        self._print_workspace_limits()
        self._log_initial_end_effector_orientation()
        if self.motion_only_mode:
            _log(
                "run_demo",
                "No gripper joints were resolved. Running motion-only arm test sequence.",
                todo="Configure gripper joints and attachment logic to enable a full pick-and-place cycle.",
            )

    def _log_state(self, state: DemoState) -> None:
        self.state = state
        self.state_start_time = time.time()
        print(f"[demo] state -> {state.name}")

    def _elapsed(self) -> float:
        return time.time() - self.state_start_time

    def _compute_wrist_pose(self, grasp_center_pose: Pose) -> Pose:
        return compose_pose(grasp_center_pose, self.grasp_to_wrist)

    def _derive_workspace_limits(self) -> WorkspaceLimits:
        joint_origins = parse_joint_origins(self.training_config.urdf_path)
        arm_joint_origins = [joint_origins[joint_name] for joint_name in self.training_config.arm_joints]

        shoulder_position = np.array(arm_joint_origins[0], dtype=np.float64)
        chain_reach = float(
            sum(np.linalg.norm(origin) for origin in arm_joint_origins[1:])
            + np.linalg.norm(self.tuning.wrist_to_grasp_translation)
        )
        radial_max = max(0.32, 0.62 * chain_reach)
        radial_min = max(0.12, 0.18 * chain_reach)

        table_clearance = self.scene.scene_config.table_height + (self.scene.scene_config.brick_scale[2] / 2.0) + 0.01
        x_min = shoulder_position[0] - 0.18
        x_max = shoulder_position[0] + 0.12
        y_min = shoulder_position[1] + 0.14
        y_max = shoulder_position[1] + min(radial_max * 0.95, 0.46)
        z_min = table_clearance
        z_max = shoulder_position[2] + 0.10

        return WorkspaceLimits(
            shoulder_position=shoulder_position,
            radial_min=radial_min,
            radial_max=radial_max,
            x_min=float(x_min),
            x_max=float(x_max),
            y_min=float(y_min),
            y_max=float(y_max),
            z_min=float(z_min),
            z_max=float(z_max),
        )

    def _log_initial_end_effector_orientation(self) -> None:
        ee_pose = self.scene.get_end_effector_pose()
        ee_forward = np.array(ee_pose.rotation[:, 2], dtype=np.float64)
        _log(
            "init_orientation",
            "End-effector forward direction at initialization.",
            ee_forward=np.round(ee_forward, 4).tolist(),
        )
        if float(ee_forward[2]) > 0.0:
            print("[warning] EE orientation likely flipped")

    def _current_target_pose(self) -> Pose | None:
        if self.plan is None:
            return None
        target_by_state = {
            DemoState.MOVE_TO_PREGRASP: self.plan.pregrasp_pose,
            DemoState.MOVE_TO_GRASP: self.plan.grasp_pose,
            DemoState.CLOSE_GRIPPER: self.plan.grasp_pose,
            DemoState.WAIT_FOR_ATTACH: self.plan.grasp_pose,
            DemoState.LIFT: self.plan.lift_pose,
            DemoState.MOVE_TO_PLACE_PREGRASP: self.plan.place_pregrasp_pose,
            DemoState.MOVE_TO_PLACE: self.plan.place_pose,
            DemoState.OPEN_GRIPPER: self.plan.place_pose,
            DemoState.RETRACT: self.plan.place_pregrasp_pose,
            DemoState.DONE: self.plan.lift_pose if not self.enable_place else self.plan.place_pregrasp_pose,
        }
        return target_by_state.get(self.state)

    def _print_workspace_limits(self) -> None:
        print(
            "[demo] workspace limits",
            {
                "shoulder_position": np.round(self.workspace.shoulder_position, 4).tolist(),
                "radial_min": round(self.workspace.radial_min, 4),
                "radial_max": round(self.workspace.radial_max, 4),
                "x_range": [round(self.workspace.x_min, 4), round(self.workspace.x_max, 4)],
                "y_range": [round(self.workspace.y_min, 4), round(self.workspace.y_max, 4)],
                "z_range": [round(self.workspace.z_min, 4), round(self.workspace.z_max, 4)],
            },
        )

    def _clamp_position_to_workspace(self, position: np.ndarray, label: str) -> np.ndarray:
        clamped = np.array(position, dtype=np.float64)
        clamped[0] = np.clip(clamped[0], self.workspace.x_min, self.workspace.x_max)
        clamped[1] = np.clip(clamped[1], self.workspace.y_min, self.workspace.y_max)
        clamped[2] = np.clip(clamped[2], self.workspace.z_min, self.workspace.z_max)

        shoulder_to_target = clamped - self.workspace.shoulder_position
        radius = float(np.linalg.norm(shoulder_to_target))
        if radius > 1e-6:
            if radius < self.workspace.radial_min:
                clamped = self.workspace.shoulder_position + shoulder_to_target * (self.workspace.radial_min / radius)
            elif radius > self.workspace.radial_max:
                clamped = self.workspace.shoulder_position + shoulder_to_target * (self.workspace.radial_max / radius)

        clamped[0] = np.clip(clamped[0], self.workspace.x_min, self.workspace.x_max)
        clamped[1] = np.clip(clamped[1], self.workspace.y_min, self.workspace.y_max)
        clamped[2] = np.clip(clamped[2], self.workspace.z_min, self.workspace.z_max)

        print(
            "[demo] workspace clamp",
            {
                "label": label,
                "input": np.round(position, 4).tolist(),
                "clamped": np.round(clamped, 4).tolist(),
            },
        )
        return clamped

    def _ensure_downward_orientation(self, rotation: np.ndarray) -> np.ndarray:
        ee_z_axis = np.array(rotation[:, 2], dtype=np.float64)
        print("[debug] ee z-axis:", np.round(ee_z_axis, 4).tolist())
        if float(ee_z_axis[2]) > 0.0:
            flipped_rotation = rotation_matrix_from_axes(-rotation[:, 0], rotation[:, 1], -rotation[:, 2])
            print("[warning] EE orientation likely flipped; using downward-facing alternative.")
            print("[debug] ee z-axis:", np.round(np.array(flipped_rotation[:, 2], dtype=np.float64), 4).tolist())
            return flipped_rotation
        return rotation

    def _build_grasp_rotation_candidates(self) -> list[np.ndarray]:
        approach_axis = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        lateral_axes = [
            np.array([1.0, 0.0, 0.0], dtype=np.float64),
            np.array([-1.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, -1.0, 0.0], dtype=np.float64),
        ]

        rotations = []
        for x_axis in lateral_axes:
            z_axis = np.cross(x_axis, approach_axis)
            if np.linalg.norm(z_axis) < 1e-6:
                continue
            candidate_rotation = rotation_matrix_from_axes(x_axis, approach_axis, z_axis)
            rotations.append(self._ensure_downward_orientation(candidate_rotation))
        return rotations

    def _solve_pose_candidates(
        self,
        pose_candidates: list[Pose],
        label: str,
        warm_start: np.ndarray,
        extra_seed: np.ndarray | None = None,
    ) -> tuple[Pose, np.ndarray]:
        for index, pose in enumerate(pose_candidates, start=1):
            solution, success = self.scene.solve_ik(pose, warm_start=warm_start, extra_seed=extra_seed)
            print(
                "[demo] IK",
                {
                    "label": label,
                    "candidate": index,
                    "success": bool(success),
                    "target_position": np.round(pose.position, 4).tolist(),
                    "target_quat_wxyz": np.round(pose.quaternion_wxyz, 4).tolist(),
                },
            )
            if success and solution is not None:
                return pose, solution
        raise RuntimeError(f"Failed to solve IK for {label}.")

    def _build_geometry_plan(self) -> GeometryPlan:
        brick_pose = self.scene.get_brick_pose()
        brick_center = self._clamp_position_to_workspace(brick_pose.position, "brick")
        if np.linalg.norm(brick_center - brick_pose.position) > 1e-6:
            self.scene.brick.set_world_pose(position=brick_center.astype(np.float32))
            self.scene.step_world(steps=2)
            brick_pose = self.scene.get_brick_pose()
            brick_center = brick_pose.position
        place_center = np.array(self.scene.scene_config.place_position, dtype=np.float64)
        place_center[2] = self.scene.scene_config.table_height + self.scene.scene_config.brick_scale[2] / 2.0
        place_center = self._clamp_position_to_workspace(place_center, "place")

        home_arm = _home_arm_configuration(self.training_config.arm_joints)

        for candidate_index, rotation in enumerate(self._build_grasp_rotation_candidates(), start=1):
            try:
                grasp_center = Pose(position=brick_center, rotation=rotation)
                pregrasp_center = Pose(
                    position=self._clamp_position_to_workspace(
                        brick_center + np.array([0.0, 0.0, self.tuning.pregrasp_height], dtype=np.float64),
                        "pregrasp",
                    ),
                    rotation=rotation,
                )
                lift_center = Pose(
                    position=self._clamp_position_to_workspace(
                        brick_center + np.array([0.0, 0.0, self.tuning.lift_height], dtype=np.float64),
                        "lift",
                    ),
                    rotation=rotation,
                )
                place_center_pose = (
                    Pose(position=self._clamp_position_to_workspace(place_center, "place_grasp"), rotation=rotation)
                    if self.enable_place
                    else None
                )
                place_pregrasp_center = (
                    Pose(
                        position=self._clamp_position_to_workspace(
                            place_center + np.array([0.0, 0.0, self.tuning.place_height], dtype=np.float64),
                            "place_pregrasp",
                        ),
                        rotation=rotation,
                    )
                    if self.enable_place
                    else None
                )

                grasp_pose = self._compute_wrist_pose(grasp_center)
                pregrasp_pose = self._compute_wrist_pose(pregrasp_center)
                lift_pose = self._compute_wrist_pose(lift_center)
                place_pose = None if place_center_pose is None else self._compute_wrist_pose(place_center_pose)
                place_pregrasp_pose = (
                    None if place_pregrasp_center is None else self._compute_wrist_pose(place_pregrasp_center)
                )

                joint_targets: dict[DemoState, np.ndarray] = {}
                _, joint_targets[DemoState.MOVE_TO_PREGRASP] = self._solve_pose_candidates(
                    [pregrasp_pose],
                    f"pregrasp_candidate_{candidate_index}",
                    warm_start=home_arm,
                )
                _, joint_targets[DemoState.MOVE_TO_GRASP] = self._solve_pose_candidates(
                    [grasp_pose],
                    f"grasp_candidate_{candidate_index}",
                    warm_start=joint_targets[DemoState.MOVE_TO_PREGRASP],
                    extra_seed=home_arm,
                )
                _, joint_targets[DemoState.LIFT] = self._solve_pose_candidates(
                    [lift_pose],
                    f"lift_candidate_{candidate_index}",
                    warm_start=joint_targets[DemoState.MOVE_TO_GRASP],
                    extra_seed=joint_targets[DemoState.MOVE_TO_PREGRASP],
                )

                if self.enable_place and place_pregrasp_pose is not None and place_pose is not None:
                    _, joint_targets[DemoState.MOVE_TO_PLACE_PREGRASP] = self._solve_pose_candidates(
                        [place_pregrasp_pose],
                        f"place_pregrasp_candidate_{candidate_index}",
                        warm_start=joint_targets[DemoState.LIFT],
                        extra_seed=joint_targets[DemoState.MOVE_TO_GRASP],
                    )
                    _, joint_targets[DemoState.MOVE_TO_PLACE] = self._solve_pose_candidates(
                        [place_pose],
                        f"place_candidate_{candidate_index}",
                        warm_start=joint_targets[DemoState.MOVE_TO_PLACE_PREGRASP],
                        extra_seed=joint_targets[DemoState.LIFT],
                    )
                    joint_targets[DemoState.RETRACT] = joint_targets[DemoState.MOVE_TO_PLACE_PREGRASP].copy()

                plan = GeometryPlan(
                    brick_pose=brick_pose,
                    grasp_pose=grasp_pose,
                    pregrasp_pose=pregrasp_pose,
                    lift_pose=lift_pose,
                    place_pose=place_pose,
                    place_pregrasp_pose=place_pregrasp_pose,
                    joint_targets=joint_targets,
                )
                self._ik_targets = joint_targets
                print("[demo] selected grasp candidate", candidate_index)
                return plan
            except RuntimeError as exc:
                print("[demo] candidate rejected", {"candidate": candidate_index, "reason": str(exc)})

        raise RuntimeError("Failed to generate a consistent IK path for the demo.")

    def _arm_reached(self, target: np.ndarray) -> bool:
        current = self.scene.current_arm_positions()
        return float(np.max(np.abs(current - target))) < self.tuning.arm_tolerance

    def _pose_reached(self, target_pose: Pose | None, position_tolerance: float = 0.05) -> bool:
        if target_pose is None:
            return False
        ee_pose = self.scene.get_end_effector_pose()
        return float(np.linalg.norm(ee_pose.position - target_pose.position)) < position_tolerance

    def _state_target_reached(self, state: DemoState) -> bool:
        state_targets = {
            DemoState.MOVE_TO_PREGRASP: self.plan.pregrasp_pose if self.plan is not None else None,
            DemoState.MOVE_TO_GRASP: self.plan.grasp_pose if self.plan is not None else None,
            DemoState.LIFT: self.plan.lift_pose if self.plan is not None else None,
            DemoState.MOVE_TO_PLACE_PREGRASP: self.plan.place_pregrasp_pose if self.plan is not None else None,
            DemoState.MOVE_TO_PLACE: self.plan.place_pose if self.plan is not None else None,
            DemoState.RETRACT: self.plan.place_pregrasp_pose if self.plan is not None else None,
        }
        return self._arm_reached(self._ik_targets[state]) or self._pose_reached(state_targets.get(state))

    def _gripper_reached(self, target: np.ndarray) -> bool:
        current = self.scene.current_gripper_positions()
        if current.size == 0 and target.size == 0:
            return True
        return float(np.max(np.abs(current - target))) < self.tuning.gripper_tolerance

    def _update_attached_brick(self) -> None:
        if not self.attached:
            return
        wrist_pose = self.scene.get_end_effector_pose()
        grasp_center_pose = compose_pose(wrist_pose, self.wrist_to_grasp)
        self.scene.brick.set_world_pose(position=grasp_center_pose.position.astype(np.float32))
        self.scene.brick.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.scene.brick.set_angular_velocity(np.zeros(3, dtype=np.float32))

    def _reset_demo(self) -> None:
        self.attached = False
        self.demo_start_time = time.time()
        self.scene.reset_scene()
        set_initial_joint_positions(self.scene)
        self.plan = self._build_geometry_plan()
        self._print_debug_snapshot(force=True)

    def _print_debug_snapshot(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_debug_time < 1.0:
            return
        self._last_debug_time = now

        base_pose = self.scene.get_base_pose()
        ee_pose = self.scene.get_end_effector_pose()
        brick_pose = self.scene.get_brick_pose()
        pregrasp = self.plan.pregrasp_pose if self.plan is not None else None
        grasp = self.plan.grasp_pose if self.plan is not None else None
        target_pose = self._current_target_pose()
        position_error = None
        if target_pose is not None:
            position_error = float(np.linalg.norm(ee_pose.position - target_pose.position))
        print(
            "[demo] debug",
            {
                "state": self.state.name,
                "robot_base_pose": np.round(base_pose.position, 4).tolist(),
                "current_ee_position": np.round(ee_pose.position, 4).tolist(),
                "current_ee_quaternion_wxyz": np.round(ee_pose.quaternion_wxyz, 4).tolist(),
                "brick_pose": np.round(brick_pose.position, 4).tolist(),
                "target_pregrasp_pose": None if pregrasp is None else np.round(pregrasp.position, 4).tolist(),
                "target_grasp_pose": None if grasp is None else np.round(grasp.position, 4).tolist(),
                "target_pose": None if target_pose is None else np.round(target_pose.position, 4).tolist(),
                "position_error": None if position_error is None else round(position_error, 6),
            },
        )

    def _command_state(self, state: DemoState, gripper_target: np.ndarray) -> None:
        # Motion speed tuning: move only partway toward each target on every update.
        arm_target = self._ik_targets[state]
        current_arm = self.scene.current_arm_positions()
        blended_arm_target = current_arm + (MOTION_SPEED_SCALE * (arm_target - current_arm))

        if gripper_target.size > 0:
            current_gripper = self.scene.current_gripper_positions()
            blended_gripper_target = current_gripper + (MOTION_SPEED_SCALE * (gripper_target - current_gripper))
        else:
            blended_gripper_target = gripper_target

        self.scene.apply_joint_targets(blended_arm_target, blended_gripper_target)

    def _brick_is_in_grasp_region(self) -> tuple[bool, dict[str, float]]:
        wrist_pose = self.scene.get_end_effector_pose()
        grasp_center_pose = compose_pose(wrist_pose, self.wrist_to_grasp)
        brick_pose = self.scene.get_brick_pose()
        delta = brick_pose.position - grasp_center_pose.position
        radial_xy = float(np.linalg.norm(delta[:2]))
        delta_z = abs(float(delta[2]))
        within_region = bool(
            radial_xy < self.tuning.attach_threshold_xy
            and delta_z < self.tuning.attach_threshold_z
        )
        metrics = {
            "radial_xy": radial_xy,
            "delta_z": delta_z,
            "threshold_xy": self.tuning.attach_threshold_xy,
            "threshold_z": self.tuning.attach_threshold_z,
        }
        return within_region, metrics

    def _attempt_attach_brick(self) -> bool:
        wrist_pose = self.scene.get_end_effector_pose()
        grasp_center_pose = compose_pose(wrist_pose, self.wrist_to_grasp)
        brick_pose = self.scene.get_brick_pose()
        target_pose = self._current_target_pose()
        position_error = None if target_pose is None else float(np.linalg.norm(wrist_pose.position - target_pose.position))
        in_region, metrics = self._brick_is_in_grasp_region()

        # Scripted demo fallback: if the arm has reached the grasp target closely, snap the brick into
        # the grasp frame so the pick-and-lift can proceed even without stable physical finger contact.
        if not in_region and position_error is not None and position_error < 0.035:
            _log(
                "attach",
                "Snapping brick into the grasp frame after reaching the grasp pose.",
                position_error=round(position_error, 6),
                metrics={key: round(value, 6) for key, value in metrics.items()},
            )
        elif not in_region:
            _log(
                "attach",
                "Brick is outside the grasp region.",
                position_error=None if position_error is None else round(position_error, 6),
                metrics={key: round(value, 6) for key, value in metrics.items()},
            )
            return False

        self.scene.brick.set_world_pose(position=grasp_center_pose.position.astype(np.float32))
        self.scene.brick.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.scene.brick.set_angular_velocity(np.zeros(3, dtype=np.float32))
        self.attached = True
        self._update_attached_brick()
        return True

    def run(self) -> None:
        self._log_state(DemoState.RESET)
        while self.scene._app.is_running():
            if time.time() - self.demo_start_time >= MAX_DEMO_SECONDS:
                _log(
                    "run_demo",
                    "Reached demo timeout. Saving logs and stopping simulation.",
                    max_demo_seconds=MAX_DEMO_SECONDS,
                )
                if not self._trajectory_saved:
                    self.trajectory_recorder.save()
                    self._trajectory_saved = True
                break

            self._print_debug_snapshot()

            ee_pose = self.scene.get_end_effector_pose()
            self.trajectory_recorder.record(
                timestamp=time.time(),
                ee_pose=ee_pose,
                joint_positions=np.array(self.scene.articulation.get_joint_positions(), dtype=np.float64),
                target_pose=self._current_target_pose(),
            )

            if self.state == DemoState.RESET:
                self._reset_demo()
                self._log_state(DemoState.MOVE_TO_PREGRASP)

            elif self.state == DemoState.MOVE_TO_PREGRASP:
                self._command_state(DemoState.MOVE_TO_PREGRASP, self.open_gripper)
                if self._state_target_reached(DemoState.MOVE_TO_PREGRASP):
                    self._log_state(DemoState.MOVE_TO_GRASP)

            elif self.state == DemoState.MOVE_TO_GRASP:
                self._command_state(DemoState.MOVE_TO_GRASP, self.open_gripper)
                if self._state_target_reached(DemoState.MOVE_TO_GRASP):
                    if self.motion_only_mode:
                        _log(
                            "run_demo",
                            "Reached grasp pose in motion-only mode.",
                            todo="Enable gripper control to close on the brick and attach it for lift/place.",
                        )
                        self._log_state(DemoState.LIFT)
                    else:
                        self._log_state(DemoState.CLOSE_GRIPPER)

            elif self.state == DemoState.CLOSE_GRIPPER:
                self._command_state(DemoState.MOVE_TO_GRASP, self.closed_gripper)
                if self._elapsed() >= self.tuning.close_gripper_seconds or self._gripper_reached(self.closed_gripper):
                    self._log_state(DemoState.WAIT_FOR_ATTACH)

            elif self.state == DemoState.WAIT_FOR_ATTACH:
                self._command_state(DemoState.MOVE_TO_GRASP, self.closed_gripper)
                if self._elapsed() >= self.tuning.attach_wait_seconds:
                    attach_success = self._attempt_attach_brick()
                    print("[demo] attach", {"success": attach_success})
                    if not attach_success:
                        raise RuntimeError("Brick was not close enough to the grasp frame to attach.")
                    self._log_state(DemoState.LIFT)

            elif self.state == DemoState.LIFT:
                self._command_state(DemoState.LIFT, self.closed_gripper)
                self._update_attached_brick()
                if self._state_target_reached(DemoState.LIFT):
                    if self.enable_place and not self.motion_only_mode:
                        self._log_state(DemoState.MOVE_TO_PLACE_PREGRASP)
                    else:
                        self._log_state(DemoState.DONE)

            elif self.state == DemoState.MOVE_TO_PLACE_PREGRASP:
                self._command_state(DemoState.MOVE_TO_PLACE_PREGRASP, self.closed_gripper)
                self._update_attached_brick()
                if self._state_target_reached(DemoState.MOVE_TO_PLACE_PREGRASP):
                    self._log_state(DemoState.MOVE_TO_PLACE)

            elif self.state == DemoState.MOVE_TO_PLACE:
                self._command_state(DemoState.MOVE_TO_PLACE, self.closed_gripper)
                self._update_attached_brick()
                if self._state_target_reached(DemoState.MOVE_TO_PLACE):
                    self._log_state(DemoState.OPEN_GRIPPER)

            elif self.state == DemoState.OPEN_GRIPPER:
                self._command_state(DemoState.MOVE_TO_PLACE, self.open_gripper)
                self._update_attached_brick()
                if self._gripper_reached(self.open_gripper):
                    self.attached = False
                    self.scene.brick.set_linear_velocity(np.zeros(3, dtype=np.float32))
                    self.scene.brick.set_angular_velocity(np.zeros(3, dtype=np.float32))
                    self._log_state(DemoState.RETRACT)

            elif self.state == DemoState.RETRACT:
                self._command_state(DemoState.RETRACT, self.open_gripper)
                if self._state_target_reached(DemoState.RETRACT):
                    self._log_state(DemoState.DONE)

            elif self.state == DemoState.DONE:
                hold_target = DemoState.RETRACT if self.enable_place and not self.motion_only_mode else DemoState.LIFT
                self._command_state(hold_target, self.closed_gripper if self.attached else self.open_gripper)
                self._update_attached_brick()
                if self._elapsed() >= self.tuning.done_hold_seconds:
                    self.demo_count += 1
                    print(f"[demo] completed demos: {self.demo_count}")
                    if not self._trajectory_saved:
                        self.trajectory_recorder.save()
                        self._trajectory_saved = True
                    if self.run_once:
                        break
                    self._log_state(DemoState.RESET)

            self.scene.step_world(steps=STEPS_PER_TARGET)
            self._update_attached_brick()


def build_arg_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Geometry-driven Isaac Sim brick-pick demo.")
    parser.add_argument("--headless", default="false")
    parser.add_argument("--run-once", default="true")
    parser.add_argument("--enable-place", default="false")
    parser.add_argument(
        "--robot-description-path",
        type=Path,
        default=repo_root / "assets" / "robot_description",
    )
    return parser


def run_demo(
    robot_description_path: Path,
    headless: bool,
    run_once: bool,
    enable_place: bool,
) -> None:
    _log(
        "run_demo",
        "Starting brick-pick demo.",
        headless=headless,
        run_once=run_once,
        enable_place=enable_place,
    )
    scene: HumanoidBrickPickDemoScene | None = None
    try:
        training_config = load_robot(robot_description_path)
        scene = create_scene(training_config=training_config, headless=headless)
        spawn_brick(scene)
        demo = BrickPickDemo(
            training_config=training_config,
            scene=scene,
            run_once=run_once,
            enable_place=enable_place,
        )
        _log(
            "run_demo",
            "Initial scene summary.",
            arm_joints=training_config.arm_joints,
            gripper_joints=training_config.gripper_joints,
        )
        demo.run()
        if not demo._trajectory_saved:
            demo.trajectory_recorder.save()
            demo._trajectory_saved = True
    except KeyboardInterrupt:
        if 'demo' in locals() and not demo._trajectory_saved:
            demo.trajectory_recorder.save()
            demo._trajectory_saved = True
        _log(
            "run_demo",
            "Interrupted by user. Saved trajectory before exit.",
        )
        raise
    except Exception as exc:
        if 'demo' in locals() and not demo._trajectory_saved:
            demo.trajectory_recorder.save()
            demo._trajectory_saved = True
        _log(
            "run_demo",
            "Demo failed during setup or execution.",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        if scene is not None:
            scene.close()


def main() -> None:
    args = build_arg_parser().parse_args()
    run_demo(
        robot_description_path=args.robot_description_path.resolve(),
        headless=_to_bool(args.headless),
        run_once=_to_bool(args.run_once),
        enable_place=_to_bool(args.enable_place),
    )


if __name__ == "__main__":
    main()
