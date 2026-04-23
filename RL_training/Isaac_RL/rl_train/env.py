from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from .config import RobotTrainingConfig, load_robot_training_config
from .isaac_import import import_urdf, verify_runtime_urdf


_SIMULATION_APP = None
# Print the joint debug table only once across all env instances.
_JOINT_DEBUG_PRINTED = False
_REACHING_DEBUG_PRINTED = False

REACHING_THRESHOLD_PHASES = {
    1: 0.18,
    2: 0.15,
    3: 0.12,
    4: 0.08,
}

DEFAULT_REACHING_BRICK_RANGE = {
    "train": {"x_min": 0.43, "x_max": 0.50, "y_min": -0.28, "y_max": -0.20},
    "eval": {"x_min": 0.41, "x_max": 0.52, "y_min": -0.30, "y_max": -0.18},
}

DEFAULT_REACHING_HOME_OVERRIDES = {
    "right_shoulder_yaw_joint": 0.35,
    "right_elbow_pitch_joint": 1.25,
    "right_wrist_pitch_joint": 1.05,
    "right_wrist_yaw_joint": 0.0,
}

DEFAULT_RIGHT_GRASP_TCP_OFFSET = np.array([0.0, 0.141, -0.0425], dtype=np.float32)


def _to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resolve_reach_threshold(
    reaching_only: bool,
    reach_distance_threshold: float | None,
    reach_threshold_phase: int | None,
) -> tuple[float, int | None]:
    if reach_distance_threshold is not None:
        return float(reach_distance_threshold), reach_threshold_phase
    if not reaching_only:
        return 0.08, None
    phase = 1 if reach_threshold_phase is None else int(reach_threshold_phase)
    if phase not in REACHING_THRESHOLD_PHASES:
        raise ValueError(
            f"Unsupported reaching threshold phase {phase}. "
            f"Expected one of {sorted(REACHING_THRESHOLD_PHASES)}."
        )
    return REACHING_THRESHOLD_PHASES[phase], phase


def _get_simulation_app(headless: bool):
    global _SIMULATION_APP
    if _SIMULATION_APP is not None:
        return _SIMULATION_APP

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    _SIMULATION_APP = SimulationApp({"headless": headless})
    return _SIMULATION_APP


@dataclass
class EpisodeTaskMetrics:
    reached_object: bool = False
    grasped_object: bool = False
    lifted_object: bool = False
    stable_hold: bool = False
    stable_hold_steps: int = 0
    max_brick_height: float = 0.0
    min_distance_to_brick: float = float("inf")
    final_distance_to_brick: float = float("inf")
    final_brick_height: float = 0.0
    cumulative_reward: float = 0.0
    # --- Per-term reward accumulators for diagnostics ---
    reward_action_penalty: float = 0.0
    reward_distance: float = 0.0
    reward_approach: float = 0.0
    reward_reach_bonus: float = 0.0
    reward_grasp_bonus: float = 0.0
    reward_success_bonus: float = 0.0
    reward_lift_bonus: float = 0.0
    reward_height_bonus: float = 0.0
    reward_velocity_penalty: float = 0.0
    action_magnitude: float = 0.0
    last_action_magnitude: float = 0.0
    action_steps: int = 0


@dataclass(frozen=True)
class TaskState:
    distance_to_brick: float
    brick_height: float
    gripper_closed_error: float
    reached_object: bool
    grasped_object: bool
    lifted_object: bool
    stable_hold: bool
    success: bool


class HumanoidBrickPickEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        training_config: RobotTrainingConfig | None = None,
        repo_root: Path | None = None,
        robot_description_path: Path | None = None,
        arm_joints: str | None = None,
        gripper_joints: str | None = None,
        end_effector_link: str | None = None,
        control_dt: float = 0.2,
        physics_dt: float = 1.0 / 60.0,
        max_steps: int = 200,          # Increased from 150 — gives PPO more time to learn reaching
        headless: bool = True,
        evaluation: bool = False,
        reaching_only: bool = False,
        reach_distance_threshold: float | None = None,
        reach_threshold_phase: int | None = None,
        use_grasp_tcp: str | bool | None = None,
        arm_action_scale: float | None = None,
        reaching_brick_range: dict[str, float] | None = None,
        reaching_home_overrides: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        self.training_config = training_config or load_robot_training_config(
            repo_root=repo_root,
            robot_description_path=robot_description_path,
            arm_joints=arm_joints,
            gripper_joints=gripper_joints,
            end_effector_link=end_effector_link,
        )
        self.control_dt = control_dt
        self.physics_dt = physics_dt
        self.max_steps = max_steps
        self.headless = _to_bool(headless)
        self.evaluation = evaluation
        self.reaching_only = _to_bool(reaching_only)
        if use_grasp_tcp is None:
            self.use_grasp_tcp = self.reaching_only
        else:
            self.use_grasp_tcp = _to_bool(use_grasp_tcp)
        self.reach_threshold_phase = reach_threshold_phase
        self.current_step = 0
        self._last_distance = None
        self._task_metrics = EpisodeTaskMetrics()

        self.reach_distance_threshold, self.reach_threshold_phase = _resolve_reach_threshold(
            self.reaching_only,
            reach_distance_threshold,
            reach_threshold_phase,
        )
        self.grasp_distance_threshold = 0.06
        self.gripper_closed_threshold = 0.012
        self.lift_height_threshold = 0.08
        self.stable_hold_steps_required = 5
        self.arm_action_scale = float(arm_action_scale) if arm_action_scale is not None else (0.03 if self.reaching_only else 0.035)
        self.reaching_brick_range = reaching_brick_range
        self.reaching_home_overrides = (
            dict(DEFAULT_REACHING_HOME_OVERRIDES)
            if self.reaching_only and reaching_home_overrides is None
            else (reaching_home_overrides or {})
        )
        self.end_effector_local_offset = (
            DEFAULT_RIGHT_GRASP_TCP_OFFSET.copy()
            if self.use_grasp_tcp and self.training_config.end_effector_link == "right_wrist_yaw_link"
            else np.zeros(3, dtype=np.float32)
        )
        self.end_effector_reference_name = (
            f"{self.training_config.end_effector_link}+right_grasp_tcp_offset"
            if bool(np.any(self.end_effector_local_offset))
            else self.training_config.end_effector_link
        )

        self._app = _get_simulation_app(self.headless)
        self._load_isaac_modules()
        self._build_scene()
        self.arm_lower, self.arm_upper = self._build_arm_bounds()

        gripper_dim = len(self.training_config.gripper_joints)
        action_dim = len(self.training_config.arm_joints) + gripper_dim
        # --- Observation space (expanded) ---
        # arm_positions:       arm_dim
        # arm_velocities:      arm_dim  (scaled by 0.1 for normalization)
        # gripper_positions:   gripper_dim
        # brick_position:      3
        # ee_position:         3
        # relative_brick:      3   (brick - ee, the most important signal)
        # gripper_closed_err:  1   (scalar: how far gripper is from closed)
        obs_dim = (
            len(self.training_config.arm_joints) * 2
            + gripper_dim
            + 3
            + 3
            + 3
            + 1   # NEW: gripper closed error scalar
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _load_isaac_modules(self) -> None:
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.types import ArticulationAction
        from pxr import Gf, UsdGeom
        import omni.kit.commands

        self._World = World
        self._Articulation = Articulation
        self._DynamicCuboid = DynamicCuboid
        self._FixedCuboid = FixedCuboid
        self._get_prim_at_path = get_prim_at_path
        self._ArticulationAction = ArticulationAction
        self._Gf = Gf
        self._UsdGeom = UsdGeom
        self._omni_kit_commands = omni.kit.commands

    def _build_scene(self) -> None:
        self.world = self._World(stage_units_in_meters=1.0, physics_dt=self.physics_dt)
        self.world.scene.add_default_ground_plane()

        self.table = self.world.scene.add(
            self._FixedCuboid(
                prim_path="/World/training_table",
                name="training_table",
                position=np.array([0.48, -0.24, 0.36], dtype=np.float32),
                scale=np.array([0.70, 0.50, 0.06], dtype=np.float32),
                color=np.array([0.45, 0.32, 0.18], dtype=np.float32),
            )
        )
        self.table_height = 0.39

        print("[train] source URDF:", self.training_config.urdf_path)
        verify_runtime_urdf(self.training_config.runtime_urdf_path)
        self.robot_prim_path = import_urdf(self.training_config.runtime_urdf_path)
        self.robot_prim = self._get_prim_at_path(self.robot_prim_path)

        self.brick = self.world.scene.add(
            self._DynamicCuboid(
                prim_path="/World/toy_brick",
                name="toy_brick",
                position=np.array([0.46, -0.24, self.table_height + 0.015], dtype=np.float32),
                scale=np.array([0.06, 0.03, 0.03], dtype=np.float32),
                color=np.array([0.8, 0.1, 0.1], dtype=np.float32),
                mass=0.05,
            )
        )

        self.articulation = self._Articulation(prim_path=self.robot_prim_path, name="humanoid")
        self.world.reset()
        self.articulation.initialize()
        self._set_robot_root_pose()

        self.dof_names = list(self.articulation.dof_names)
        self.arm_indices = [self.dof_names.index(name) for name in self.training_config.arm_joints]
        self.gripper_indices = [self.dof_names.index(name) for name in self.training_config.gripper_joints]
        self.closed_gripper_positions = self._named_joint_array(self.training_config.closed_gripper_positions)
        self.open_gripper_positions = self._named_joint_array(self.training_config.open_gripper_positions)
        self.ee_prim = self._get_prim_at_path(
            f"{self.robot_prim_path}/{self.training_config.end_effector_link}"
        )

    def _apply_action(self, joint_targets: np.ndarray) -> None:
        self.articulation.apply_action(self._ArticulationAction(joint_positions=joint_targets))

    def _get_last_applied_action(self):
        if hasattr(self.articulation, "get_applied_action"):
            return self.articulation.get_applied_action()
        if hasattr(self.articulation, "get_applied_actions"):
            return self.articulation.get_applied_actions()
        return None

    def _build_arm_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        for joint_name in self.training_config.arm_joints:
            joint = self.training_config.joint_limits.get(joint_name, {})
            lower.append(float(joint.get("min_position", -3.14)))
            upper.append(float(joint.get("max_position", 3.14)))
        return np.array(lower, dtype=np.float32), np.array(upper, dtype=np.float32)

    def _clamp_joint_home_value(self, joint_name: str, value: float) -> float:
        limit = self.training_config.joint_limits.get(joint_name)
        if limit is None:
            return float(value)
        lower = float(limit.get("min_position", -np.inf))
        upper = float(limit.get("max_position", np.inf))
        return float(np.clip(value, lower, upper))

    def _reset_task_metrics(self) -> None:
        self._task_metrics = EpisodeTaskMetrics()

    def _set_robot_root_pose(self) -> None:
        root_position = np.array(self.training_config.robot_root_position, dtype=np.float32)
        if hasattr(self.articulation, "set_world_pose"):
            self.articulation.set_world_pose(position=root_position)
            return
        self._UsdGeom.XformCommonAPI(self.robot_prim).SetTranslate(tuple(float(v) for v in root_position))

    def _named_joint_array(self, named_positions: dict[str, float]) -> np.ndarray:
        return np.array(
            [float(named_positions.get(joint_name, 0.0)) for joint_name in self.training_config.gripper_joints],
            dtype=np.float32,
        )

    def _gripper_closed_error(self, gripper_positions: np.ndarray) -> float:
        if gripper_positions.size == 0:
            return 0.0
        return float(np.mean(np.abs(gripper_positions - self.closed_gripper_positions)))

    def _sample_brick_xy(self) -> tuple[float, float]:
        if self.reaching_only:
            if self.reaching_brick_range is not None:
                sample_range = self.reaching_brick_range
            else:
                sample_range = (
                    DEFAULT_REACHING_BRICK_RANGE["eval"]
                    if self.evaluation
                    else DEFAULT_REACHING_BRICK_RANGE["train"]
                )
        else:
            sample_range = self.training_config.eval_brick_range if self.evaluation else self.training_config.train_brick_range
        x = self.np_random.uniform(sample_range["x_min"], sample_range["x_max"])
        y = self.np_random.uniform(sample_range["y_min"], sample_range["y_max"])
        return float(x), float(y)

    def _set_robot_home(self) -> None:
        """Set the robot to the raised pre-grasp home pose defined in config."""
        joint_positions = np.zeros(len(self.dof_names), dtype=np.float32)
        joint_velocities = np.zeros(len(self.dof_names), dtype=np.float32)
        for joint_name, value in self.training_config.home_joint_positions.items():
            if joint_name in self.dof_names:
                joint_positions[self.dof_names.index(joint_name)] = self._clamp_joint_home_value(joint_name, value)
        for joint_name, value in self.reaching_home_overrides.items():
            if joint_name in self.dof_names:
                joint_positions[self.dof_names.index(joint_name)] = self._clamp_joint_home_value(joint_name, value)
        for joint_index, value in zip(self.gripper_indices, self.open_gripper_positions):
            joint_positions[joint_index] = float(value)
        self._set_robot_root_pose()
        self.articulation.set_joint_positions(joint_positions)
        self.articulation.set_joint_velocities(joint_velocities)
        self._apply_action(joint_positions)

    def _print_joint_debug_table(self) -> None:
        """Print a one-time debug table of joint names, indices, limits, and reset values."""
        global _JOINT_DEBUG_PRINTED
        if _JOINT_DEBUG_PRINTED:
            return
        _JOINT_DEBUG_PRINTED = True

        current_positions = np.array(self.articulation.get_joint_positions(), dtype=np.float32)
        print("\n" + "=" * 90)
        print("[train] JOINT DEBUG TABLE — initial reset pose")
        print(
            f"{'idx':>4}  {'joint_name':<35}  {'urdf_lo':>8}  {'urdf_hi':>8}  "
            f"{'train_lo':>8}  {'train_hi':>8}  {'reset_val':>10}  {'source':<34}"
        )
        print("-" * 118)
        for idx, name in enumerate(self.dof_names):
            limits = self.training_config.joint_limits.get(name, {})
            urdf_lo = limits.get("urdf_min_position", limits.get("min_position", float("nan")))
            urdf_hi = limits.get("urdf_max_position", limits.get("max_position", float("nan")))
            lo = limits.get("min_position", float("nan"))
            hi = limits.get("max_position", float("nan"))
            source = limits.get("limit_source", "urdf")
            val = current_positions[idx]
            marker = " <-- ARM" if name in self.training_config.arm_joints else ""
            marker = " <-- GRIP" if name in self.training_config.gripper_joints else marker
            print(
                f"{idx:4d}  {name:<35}  {urdf_lo:8.4f}  {urdf_hi:8.4f}  "
                f"{lo:8.4f}  {hi:8.4f}  {val:10.4f}  {source:<34}{marker}"
            )
        print("=" * 90)

        # Also print the home_joint_positions dict for quick reference
        print("[train] home_joint_positions:", self.training_config.home_joint_positions)
        if self.reaching_home_overrides:
            print("[train] reaching_home_overrides:", self.reaching_home_overrides)
        print("[train] arm_joints:", self.training_config.arm_joints)
        print("[train] gripper_joints:", self.training_config.gripper_joints)

        # Print EE position at reset
        ee_pos = self._get_end_effector_position()
        print(f"[train] EE position at reset: {ee_pos.tolist()}")
        print(f"[train] table_height: {self.table_height}")
        print()

    def _print_reaching_debug_config(self) -> None:
        global _REACHING_DEBUG_PRINTED
        if _REACHING_DEBUG_PRINTED:
            return
        _REACHING_DEBUG_PRINTED = True
        print("[train] reaching debug config:", {
            "reaching_only": self.reaching_only,
            "reach_distance_threshold": self.reach_distance_threshold,
            "reach_threshold_phase": self.reach_threshold_phase,
            "end_effector_link": self.training_config.end_effector_link,
            "end_effector_reference": self.end_effector_reference_name,
            "end_effector_local_offset": self.end_effector_local_offset.tolist(),
            "arm_action_scale": self.arm_action_scale,
            "brick_range": (
                self.reaching_brick_range
                if self.reaching_brick_range is not None
                else (
                    DEFAULT_REACHING_BRICK_RANGE["eval"]
                    if self.evaluation
                    else DEFAULT_REACHING_BRICK_RANGE["train"]
                )
            ) if self.reaching_only else (
                self.training_config.eval_brick_range
                if self.evaluation
                else self.training_config.train_brick_range
            ),
        })

    def _step_world(self, steps: int = 1) -> None:
        for _ in range(max(1, steps)):
            self.world.step(render=not self.headless)

    def _get_end_effector_position(self) -> np.ndarray:
        ee_tf = self._UsdGeom.Xformable(self.ee_prim).ComputeLocalToWorldTransform(0.0)
        if bool(np.any(self.end_effector_local_offset)):
            offset = self.end_effector_local_offset
            ee_translation = ee_tf.Transform(self._Gf.Vec3d(float(offset[0]), float(offset[1]), float(offset[2])))
        else:
            ee_translation = ee_tf.ExtractTranslation()
        return np.array(
            [float(ee_translation[0]), float(ee_translation[1]), float(ee_translation[2])],
            dtype=np.float32,
        )

    def _get_observation(self) -> dict[str, np.ndarray | bool]:
        joint_positions = np.array(self.articulation.get_joint_positions(), dtype=np.float32)
        joint_velocities = np.array(self.articulation.get_joint_velocities(), dtype=np.float32)
        arm_positions = joint_positions[self.arm_indices]
        arm_velocities = joint_velocities[self.arm_indices]
        gripper_positions = joint_positions[self.gripper_indices]
        brick_position = np.array(self.brick.get_world_pose()[0], dtype=np.float32)
        ee_position = self._get_end_effector_position()
        gripper_closed_error = self._gripper_closed_error(gripper_positions)
        return {
            "arm_positions": arm_positions,
            "arm_velocities": arm_velocities,
            "gripper_positions": gripper_positions,
            "brick_position": brick_position,
            "ee_position": ee_position,
            "gripper_closed_error": gripper_closed_error,
        }

    def _flatten_obs(self, observation: dict[str, np.ndarray | bool]) -> np.ndarray:
        relative_brick = observation["brick_position"] - observation["ee_position"]
        return np.concatenate(
            [
                observation["arm_positions"],
                observation["arm_velocities"] * 0.1,  # Scale velocities down for normalization
                observation["gripper_positions"],
                observation["brick_position"],
                observation["ee_position"],
                relative_brick,
                np.array([observation["gripper_closed_error"]], dtype=np.float32),  # NEW
            ]
        ).astype(np.float32)

    def _compute_task_state(self, observation: dict[str, np.ndarray | bool]) -> TaskState:
        distance = float(np.linalg.norm(observation["brick_position"] - observation["ee_position"]))
        brick_height = float(observation["brick_position"][2])
        gripper_closed_error = float(observation["gripper_closed_error"])

        # Intended progression: reach -> grasp -> lift -> stable hold -> success.
        # Later stages are gated by earlier stages so rewards and metrics cannot
        # report a lift/success from merely bumping the brick or closing far away.
        reached_object = distance < self.reach_distance_threshold
        grasped_object = (
            reached_object
            and distance < self.grasp_distance_threshold
            and gripper_closed_error < self.gripper_closed_threshold
        )
        lifted_object = (
            grasped_object
            and brick_height > self.table_height + self.lift_height_threshold
        )

        stable_hold_steps = self._task_metrics.stable_hold_steps + 1 if lifted_object else 0
        stable_hold = stable_hold_steps >= self.stable_hold_steps_required
        success = reached_object if self.reaching_only else stable_hold
        return TaskState(
            distance_to_brick=distance,
            brick_height=brick_height,
            gripper_closed_error=gripper_closed_error,
            reached_object=reached_object,
            grasped_object=grasped_object,
            lifted_object=lifted_object,
            stable_hold=stable_hold,
            success=success,
        )

    def _update_task_metrics(self, state: TaskState) -> None:
        self._task_metrics.reached_object |= state.reached_object
        self._task_metrics.grasped_object |= state.grasped_object
        self._task_metrics.lifted_object |= state.lifted_object
        self._task_metrics.stable_hold_steps = (
            self._task_metrics.stable_hold_steps + 1 if state.lifted_object else 0
        )
        self._task_metrics.stable_hold |= (
            self._task_metrics.stable_hold_steps >= self.stable_hold_steps_required
        )
        self._task_metrics.max_brick_height = max(self._task_metrics.max_brick_height, state.brick_height)
        self._task_metrics.min_distance_to_brick = min(
            self._task_metrics.min_distance_to_brick, state.distance_to_brick
        )
        self._task_metrics.final_distance_to_brick = state.distance_to_brick
        self._task_metrics.final_brick_height = state.brick_height

    def _build_task_info(self, state: TaskState) -> dict[str, float | bool]:
        mean_action_magnitude = (
            self._task_metrics.action_magnitude / self._task_metrics.action_steps
            if self._task_metrics.action_steps > 0
            else 0.0
        )

        return {
            "distance_to_brick": state.distance_to_brick,
            "brick_height": state.brick_height,
            "final_distance_to_brick": self._task_metrics.final_distance_to_brick,
            "final_brick_height": self._task_metrics.final_brick_height,
            "reached_threshold": self.reach_distance_threshold,
            "reach_threshold_phase": float(self.reach_threshold_phase or 0),
            "min_episode_distance": self._task_metrics.min_distance_to_brick,
            "reached_object": self._task_metrics.reached_object,
            "grasped_object": self._task_metrics.grasped_object,
            "lifted_object": self._task_metrics.lifted_object,
            "stable_hold": self._task_metrics.stable_hold,
            "stable_hold_steps": float(self._task_metrics.stable_hold_steps),
            "max_brick_height": self._task_metrics.max_brick_height,
            "min_distance_to_brick": self._task_metrics.min_distance_to_brick,
            "is_success": self._task_metrics.reached_object if self.reaching_only else self._task_metrics.stable_hold,
            # --- Per-term reward diagnostics ---
            "reward_action_penalty": self._task_metrics.reward_action_penalty,
            "reward_distance": self._task_metrics.reward_distance,
            "reward_approach": self._task_metrics.reward_approach,
            "reward_reach_bonus": self._task_metrics.reward_reach_bonus,
            "reward_grasp_bonus": self._task_metrics.reward_grasp_bonus,
            "reward_success_bonus": self._task_metrics.reward_success_bonus,
            "reward_lift_bonus": self._task_metrics.reward_lift_bonus,
            "reward_height_bonus": self._task_metrics.reward_height_bonus,
            "reward_velocity_penalty": self._task_metrics.reward_velocity_penalty,
            "action_magnitude": mean_action_magnitude,
            "last_action_magnitude": self._task_metrics.last_action_magnitude,
            "current_reached_object": state.reached_object,
            "current_grasped_object": state.grasped_object,
            "current_lifted_object": state.lifted_object,
            "current_stable_hold": state.stable_hold,
            "current_success": state.success,
        }

    def _compute_reward(
        self,
        observation: dict[str, np.ndarray | bool],
        action: np.ndarray,
    ) -> tuple[float, bool, dict[str, float | bool]]:
        """Shaped reward using the same event definitions reported to TensorBoard."""
        state = self._compute_task_state(observation)
        arm_dim = len(self.training_config.arm_joints)
        action_magnitude = float(np.linalg.norm(action[:arm_dim]))
        self._task_metrics.action_magnitude += action_magnitude
        self._task_metrics.last_action_magnitude = action_magnitude
        self._task_metrics.action_steps += 1

        action_penalty = -0.08 * action_magnitude
        reward = action_penalty
        self._task_metrics.reward_action_penalty += action_penalty

        vel_penalty = -0.01 * float(np.linalg.norm(observation["arm_velocities"]))
        reward += vel_penalty
        self._task_metrics.reward_velocity_penalty += vel_penalty

        dist_term = -2.0 * state.distance_to_brick
        reward += dist_term
        self._task_metrics.reward_distance += dist_term

        if self._last_distance is not None:
            approach_term = 6.0 * (self._last_distance - state.distance_to_brick)
            reward += approach_term
            self._task_metrics.reward_approach += approach_term
        self._last_distance = state.distance_to_brick

        if self.reaching_only:
            staged_bonus = 0.0
            for threshold, bonus in ((0.25, 0.15), (0.18, 0.35), (0.12, 0.60)):
                if state.distance_to_brick < threshold:
                    staged_bonus += bonus
            if staged_bonus:
                reward += staged_bonus
                self._task_metrics.reward_reach_bonus += staged_bonus
            if state.success:
                reward += 10.0
                self._task_metrics.reward_reach_bonus += 5.0
                self._task_metrics.reward_success_bonus += 5.0
            self._update_task_metrics(state)
            info = self._build_task_info(state)
            return reward, bool(state.success), info

        if state.reached_object:
            reach_bonus = 2.0
            reward += reach_bonus
            self._task_metrics.reward_reach_bonus += reach_bonus

        if state.grasped_object:
            grasp_bonus = 5.0
            reward += grasp_bonus
            self._task_metrics.reward_grasp_bonus += grasp_bonus

        if state.grasped_object and state.brick_height > self.table_height + 0.02:
            height_bonus = 5.0 * (state.brick_height - self.table_height)
            reward += height_bonus
            self._task_metrics.reward_height_bonus += height_bonus

        if state.lifted_object:
            lift_bonus = 15.0
            reward += lift_bonus
            self._task_metrics.reward_lift_bonus += lift_bonus

        if state.success:
            success_bonus = 25.0
            reward += success_bonus
            self._task_metrics.reward_success_bonus += success_bonus

        self._update_task_metrics(state)
        terminated = bool(state.success)
        info = self._build_task_info(state)
        return reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self._last_distance = None
        self._reset_task_metrics()

        self.world.reset()
        self.articulation.initialize()
        self._set_robot_root_pose()
        self._set_robot_home()
        x, y = self._sample_brick_xy()
        self.brick.set_world_pose(position=np.array([x, y, self.table_height + 0.01], dtype=np.float32))
        self.brick.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.brick.set_angular_velocity(np.zeros(3, dtype=np.float32))
        self._step_world(steps=10)

        # Print the joint debug table once after the first reset
        self._print_joint_debug_table()
        self._print_reaching_debug_config()

        observation = self._get_observation()
        return self._flatten_obs(observation), {}

    def step(self, action: np.ndarray):
        self.current_step += 1
        current_observation = self._get_observation()
        arm_dim = len(self.training_config.arm_joints)

        arm_delta = action[:arm_dim] * self.arm_action_scale
        arm_targets = np.clip(current_observation["arm_positions"] + arm_delta, self.arm_lower, self.arm_upper)
        gripper_action = action[arm_dim:]
        gripper_alpha = ((gripper_action + 1.0) * 0.5).astype(np.float32)
        gripper_targets = self.closed_gripper_positions + gripper_alpha * (
            self.open_gripper_positions - self.closed_gripper_positions
        )

        joint_targets = np.array(self.articulation.get_joint_positions(), dtype=np.float32)
        joint_targets[self.arm_indices] = arm_targets
        joint_targets[self.gripper_indices] = gripper_targets
        self._apply_action(joint_targets)

        sim_steps = max(1, int(round(self.control_dt / self.physics_dt)))
        self._step_world(steps=sim_steps)

        new_observation = self._get_observation()
        reward, terminated, info = self._compute_reward(new_observation, action)
        self._task_metrics.cumulative_reward += reward
        truncated = self.current_step >= self.max_steps

        # --- Termination: brick fell off table (NEW) ---
        brick_z = float(new_observation["brick_position"][2])
        if brick_z < self.table_height - 0.10:
            truncated = True
            info["brick_fell"] = True

        if terminated or truncated:
            info["episode_completed"] = True
            info["episode_reward"] = self._task_metrics.cumulative_reward
            info["episode_length"] = float(self.current_step)
        return self._flatten_obs(new_observation), reward, terminated, truncated, info

    def close(self) -> None:
        return None
