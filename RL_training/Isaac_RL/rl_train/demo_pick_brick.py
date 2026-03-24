from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
import time

import numpy as np

from .brick_pick_demo_support import HumanoidBrickPickDemoScene, parse_joint_origins
from .config import load_robot_training_config
from .pose_math import Pose, compose_pose, invert_pose, rotation_matrix_from_axes


def _to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
    close_gripper_seconds: float = 0.8
    attach_wait_seconds: float = 0.3
    done_hold_seconds: float = 2.0
    step_dt: float = 1.0 / 60.0
    pregrasp_height: float = 0.10
    lift_height: float = 0.18
    place_height: float = 0.10
    attach_threshold_xy: float = 0.03
    attach_threshold_z: float = 0.04
    wrist_to_grasp_translation: np.ndarray | None = None
    wrist_to_grasp_rotation: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.wrist_to_grasp_translation is None:
            self.wrist_to_grasp_translation = np.array([0.0, 0.141, -0.0425], dtype=np.float64)
        if self.wrist_to_grasp_rotation is None:
            self.wrist_to_grasp_rotation = np.eye(3, dtype=np.float64)


class BrickPickDemo:
    def __init__(
        self,
        robot_description_path: Path,
        headless: bool,
        run_once: bool,
        enable_place: bool,
    ) -> None:
        self.training_config = load_robot_training_config(
            repo_root=Path(__file__).resolve().parents[1],
            robot_description_path=robot_description_path,
        )
        self.scene = HumanoidBrickPickDemoScene(training_config=self.training_config, headless=headless)
        self.run_once = run_once
        self.enable_place = enable_place
        self.tuning = DemoTuning()

        self.arm_dim = len(self.training_config.arm_joints)
        self.gripper_dim = len(self.training_config.gripper_joints)
        self.open_gripper = np.array([0.03, -0.03], dtype=np.float64)[: self.gripper_dim]
        self.closed_gripper = np.array([0.0, 0.0], dtype=np.float64)[: self.gripper_dim]

        self.state = DemoState.RESET
        self.state_start_time = time.time()
        self.demo_count = 0
        self.attached = False
        self.plan: GeometryPlan | None = None
        self._ik_targets: dict[DemoState, np.ndarray] = {}
        self._last_debug_time = 0.0

        self.wrist_to_grasp = Pose(
            position=self.tuning.wrist_to_grasp_translation,
            rotation=self.tuning.wrist_to_grasp_rotation,
        )
        self.grasp_to_wrist = invert_pose(self.wrist_to_grasp)
        self.workspace = self._derive_workspace_limits()
        self._print_workspace_limits()

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
            rotations.append(rotation_matrix_from_axes(x_axis, approach_axis, z_axis))
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

        home_arm = np.zeros(self.arm_dim, dtype=np.float64)

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

    def _gripper_reached(self, target: np.ndarray) -> bool:
        current = self.scene.current_gripper_positions()
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
        self.scene.reset_scene()
        self.scene.apply_joint_targets(np.zeros(self.arm_dim, dtype=np.float64), self.open_gripper)
        self.scene.step_world(steps=10)
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
        print(
            "[demo] debug",
            {
                "state": self.state.name,
                "robot_base_pose": np.round(base_pose.position, 4).tolist(),
                "end_effector_pose": np.round(ee_pose.position, 4).tolist(),
                "brick_pose": np.round(brick_pose.position, 4).tolist(),
                "target_pregrasp_pose": None if pregrasp is None else np.round(pregrasp.position, 4).tolist(),
                "target_grasp_pose": None if grasp is None else np.round(grasp.position, 4).tolist(),
            },
        )

    def _command_state(self, state: DemoState, gripper_target: np.ndarray) -> None:
        arm_target = self._ik_targets[state]
        self.scene.apply_joint_targets(arm_target, gripper_target)

    def _brick_is_in_grasp_region(self) -> bool:
        wrist_pose = self.scene.get_end_effector_pose()
        grasp_center_pose = compose_pose(wrist_pose, self.wrist_to_grasp)
        brick_pose = self.scene.get_brick_pose()
        delta = brick_pose.position - grasp_center_pose.position
        return bool(
            np.linalg.norm(delta[:2]) < self.tuning.attach_threshold_xy
            and abs(float(delta[2])) < self.tuning.attach_threshold_z
        )

    def run(self) -> None:
        self._log_state(DemoState.RESET)
        while self.scene._app.is_running():
            self._print_debug_snapshot()

            if self.state == DemoState.RESET:
                self._reset_demo()
                self._log_state(DemoState.MOVE_TO_PREGRASP)

            elif self.state == DemoState.MOVE_TO_PREGRASP:
                self._command_state(DemoState.MOVE_TO_PREGRASP, self.open_gripper)
                if self._arm_reached(self._ik_targets[DemoState.MOVE_TO_PREGRASP]):
                    self._log_state(DemoState.MOVE_TO_GRASP)

            elif self.state == DemoState.MOVE_TO_GRASP:
                self._command_state(DemoState.MOVE_TO_GRASP, self.open_gripper)
                if self._arm_reached(self._ik_targets[DemoState.MOVE_TO_GRASP]):
                    self._log_state(DemoState.CLOSE_GRIPPER)

            elif self.state == DemoState.CLOSE_GRIPPER:
                self._command_state(DemoState.MOVE_TO_GRASP, self.closed_gripper)
                if self._elapsed() >= self.tuning.close_gripper_seconds or self._gripper_reached(self.closed_gripper):
                    self._log_state(DemoState.WAIT_FOR_ATTACH)

            elif self.state == DemoState.WAIT_FOR_ATTACH:
                self._command_state(DemoState.MOVE_TO_GRASP, self.closed_gripper)
                if self._elapsed() >= self.tuning.attach_wait_seconds:
                    self.attached = self._brick_is_in_grasp_region()
                    print("[demo] attach", {"success": self.attached})
                    if not self.attached:
                        raise RuntimeError("Brick was not in the expected grasp region at attach time.")
                    self._update_attached_brick()
                    self._log_state(DemoState.LIFT)

            elif self.state == DemoState.LIFT:
                self._command_state(DemoState.LIFT, self.closed_gripper)
                self._update_attached_brick()
                if self._arm_reached(self._ik_targets[DemoState.LIFT]):
                    if self.enable_place:
                        self._log_state(DemoState.MOVE_TO_PLACE_PREGRASP)
                    else:
                        self._log_state(DemoState.DONE)

            elif self.state == DemoState.MOVE_TO_PLACE_PREGRASP:
                self._command_state(DemoState.MOVE_TO_PLACE_PREGRASP, self.closed_gripper)
                self._update_attached_brick()
                if self._arm_reached(self._ik_targets[DemoState.MOVE_TO_PLACE_PREGRASP]):
                    self._log_state(DemoState.MOVE_TO_PLACE)

            elif self.state == DemoState.MOVE_TO_PLACE:
                self._command_state(DemoState.MOVE_TO_PLACE, self.closed_gripper)
                self._update_attached_brick()
                if self._arm_reached(self._ik_targets[DemoState.MOVE_TO_PLACE]):
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
                if self._arm_reached(self._ik_targets[DemoState.RETRACT]):
                    self._log_state(DemoState.DONE)

            elif self.state == DemoState.DONE:
                hold_target = DemoState.RETRACT if self.enable_place else DemoState.LIFT
                self._command_state(hold_target, self.closed_gripper if self.attached else self.open_gripper)
                self._update_attached_brick()
                if self._elapsed() >= self.tuning.done_hold_seconds:
                    self.demo_count += 1
                    print(f"[demo] completed demos: {self.demo_count}")
                    if self.run_once:
                        break
                    self._log_state(DemoState.RESET)

            self.scene.step_world(steps=max(1, int(round((1.0 / 30.0) / self.tuning.step_dt))))
            self._update_attached_brick()

        self.scene.close()


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


def main() -> None:
    args = build_arg_parser().parse_args()
    demo = BrickPickDemo(
        robot_description_path=args.robot_description_path.resolve(),
        headless=_to_bool(args.headless),
        run_once=_to_bool(args.run_once),
        enable_place=_to_bool(args.enable_place),
    )
    demo.run()


if __name__ == "__main__":
    main()
