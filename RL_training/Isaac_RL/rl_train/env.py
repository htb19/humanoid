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


def _to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


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
    cumulative_reward: float = 0.0


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
        max_steps: int = 150,
        headless: bool = True,
        evaluation: bool = False,
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
        self.current_step = 0
        self._last_distance = None
        self._task_metrics = EpisodeTaskMetrics()

        self._app = _get_simulation_app(self.headless)
        self._load_isaac_modules()
        self._build_scene()
        self.arm_lower, self.arm_upper = self._build_arm_bounds()

        gripper_dim = len(self.training_config.gripper_joints)
        action_dim = len(self.training_config.arm_joints) + gripper_dim
        obs_dim = (
            len(self.training_config.arm_joints) * 2
            + gripper_dim
            + 3
            + 3
            + 3
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _load_isaac_modules(self) -> None:
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.types import ArticulationAction
        from pxr import UsdGeom
        import omni.kit.commands

        self._World = World
        self._Articulation = Articulation
        self._DynamicCuboid = DynamicCuboid
        self._FixedCuboid = FixedCuboid
        self._get_prim_at_path = get_prim_at_path
        self._ArticulationAction = ArticulationAction
        self._UsdGeom = UsdGeom
        self._omni_kit_commands = omni.kit.commands

    def _build_scene(self) -> None:
        self.world = self._World(stage_units_in_meters=1.0, physics_dt=self.physics_dt)
        self.world.scene.add_default_ground_plane()

        self.table = self.world.scene.add(
            self._FixedCuboid(
                prim_path="/World/training_table",
                name="training_table",
                # In this URDF, +Y is forward and +X is lateral. The previous table pose
                # used +X as "front", which actually placed the task on the robot's right.
                position=np.array([0.0, 0.58, 0.36], dtype=np.float32),
                scale=np.array([0.80, 0.80, 0.06], dtype=np.float32),
                color=np.array([0.45, 0.32, 0.18], dtype=np.float32),
            )
        )
        self.table_height = 0.39

        print("[train] source URDF:", self.training_config.urdf_path)
        verify_runtime_urdf(self.training_config.runtime_urdf_path)
        self.robot_prim_path = import_urdf(self.training_config.runtime_urdf_path)

        self.brick = self.world.scene.add(
            self._DynamicCuboid(
                prim_path="/World/toy_brick",
                name="toy_brick",
                position=np.array([0.0, 0.58, self.table_height + 0.015], dtype=np.float32),
                scale=np.array([0.06, 0.03, 0.03], dtype=np.float32),
                color=np.array([0.8, 0.1, 0.1], dtype=np.float32),
                mass=0.05,
            )
        )

        self.articulation = self._Articulation(prim_path=self.robot_prim_path, name="humanoid")
        self.world.reset()
        self.articulation.initialize()

        self.dof_names = list(self.articulation.dof_names)
        self.arm_indices = [self.dof_names.index(name) for name in self.training_config.arm_joints]
        self.gripper_indices = [self.dof_names.index(name) for name in self.training_config.gripper_joints]
        self.ee_prim = self._get_prim_at_path(
            f"{self.robot_prim_path}/{self.training_config.end_effector_link}"
        )

    def _apply_action(self, joint_targets: np.ndarray) -> None:
        # Isaac Sim 5.x exposes `get_applied_action()` on the single-articulation API and
        # `get_applied_actions()` on articulation views. The controller wrapper can route
        # through a view path that calls the wrong method name, so for a single robot we
        # apply actions directly on the articulation primitive.
        self.articulation.apply_action(self._ArticulationAction(joint_positions=joint_targets))

    def _get_last_applied_action(self):
        # For a single robot articulation, the documented API returns the last command via
        # `get_applied_action()`. Multi-articulation views use `get_applied_actions()`.
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

    def _reset_task_metrics(self) -> None:
        self._task_metrics = EpisodeTaskMetrics()

    def _sample_brick_xy(self) -> tuple[float, float]:
        sample_range = self.training_config.eval_brick_range if self.evaluation else self.training_config.train_brick_range
        x = self.np_random.uniform(sample_range["x_min"], sample_range["x_max"])
        y = self.np_random.uniform(sample_range["y_min"], sample_range["y_max"])
        return float(x), float(y)

    def _set_robot_home(self) -> None:
        joint_positions = np.zeros(len(self.dof_names), dtype=np.float32)
        joint_velocities = np.zeros(len(self.dof_names), dtype=np.float32)
        self.articulation.set_joint_positions(joint_positions)
        self.articulation.set_joint_velocities(joint_velocities)
        self._apply_action(joint_positions)

    def _step_world(self, steps: int = 1) -> None:
        for _ in range(max(1, steps)):
            self.world.step(render=not self.headless)

    def _get_end_effector_position(self) -> np.ndarray:
        ee_tf = self._UsdGeom.Xformable(self.ee_prim).ComputeLocalToWorldTransform(0.0)
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
        horizontal_distance = np.linalg.norm(brick_position[:2] - ee_position[:2])
        grasp_success = bool(
            horizontal_distance < 0.04
            and float(np.mean(gripper_positions)) > 0.02
            and brick_position[2] > self.table_height + 0.01
        )
        return {
            "arm_positions": arm_positions,
            "arm_velocities": arm_velocities,
            "gripper_positions": gripper_positions,
            "brick_position": brick_position,
            "ee_position": ee_position,
            "grasp_success": grasp_success,
        }

    def _flatten_obs(self, observation: dict[str, np.ndarray | bool]) -> np.ndarray:
        relative_brick = observation["brick_position"] - observation["ee_position"]
        return np.concatenate(
            [
                observation["arm_positions"],
                observation["arm_velocities"],
                observation["gripper_positions"],
                observation["brick_position"],
                observation["ee_position"],
                relative_brick,
            ]
        ).astype(np.float32)

    def _build_task_info(self, distance: float, lift_height: float, grasp_success: bool) -> dict[str, float | bool]:
        reached_object = distance < 0.06
        grasped_object = bool(grasp_success or (distance < 0.04 and lift_height > self.table_height + 0.01))
        lifted_object = lift_height > self.table_height + 0.08

        self._task_metrics.reached_object |= reached_object
        self._task_metrics.grasped_object |= grasped_object
        self._task_metrics.lifted_object |= lifted_object
        self._task_metrics.max_brick_height = max(self._task_metrics.max_brick_height, lift_height)
        self._task_metrics.min_distance_to_brick = min(self._task_metrics.min_distance_to_brick, distance)

        if grasped_object and lift_height > self.table_height + 0.05:
            self._task_metrics.stable_hold_steps += 1
        else:
            self._task_metrics.stable_hold_steps = 0
        self._task_metrics.stable_hold |= self._task_metrics.stable_hold_steps >= 5

        return {
            "distance_to_brick": distance,
            "brick_height": lift_height,
            "reached_object": self._task_metrics.reached_object,
            "grasped_object": self._task_metrics.grasped_object,
            "lifted_object": self._task_metrics.lifted_object,
            "stable_hold": self._task_metrics.stable_hold,
            "stable_hold_steps": float(self._task_metrics.stable_hold_steps),
            "max_brick_height": self._task_metrics.max_brick_height,
            "min_distance_to_brick": self._task_metrics.min_distance_to_brick,
            "is_success": self._task_metrics.lifted_object,
        }

    def _compute_reward(
        self,
        observation: dict[str, np.ndarray | bool],
        action: np.ndarray,
    ) -> tuple[float, bool, dict[str, float | bool]]:
        distance = float(np.linalg.norm(observation["brick_position"] - observation["ee_position"]))
        lift_height = float(observation["brick_position"][2])
        reward = -0.1 * float(np.linalg.norm(action[: len(self.training_config.arm_joints)]))
        reward -= 2.0 * distance

        if self._last_distance is not None:
            reward += 3.0 * (self._last_distance - distance)
        self._last_distance = distance

        if distance < 0.06:
            reward += 5.0
        if distance < 0.04 and float(np.mean(observation["gripper_positions"])) > 0.02:
            reward += 10.0
        if bool(observation["grasp_success"]):
            reward += 40.0
        if lift_height > self.table_height + 0.05:
            reward += 30.0
        if lift_height > self.table_height + 0.10:
            reward += 50.0

        terminated = bool(observation["grasp_success"] and lift_height > self.table_height + 0.10)
        info = self._build_task_info(distance, lift_height, bool(observation["grasp_success"]))
        return reward, terminated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self.current_step = 0
        self._last_distance = None
        self._reset_task_metrics()

        # Reset the simulator state first so each episode starts from a clean physics state,
        # then place the robot and brick in the task-specific start configuration.
        self.world.reset()
        self.articulation.initialize()
        self._set_robot_home()
        x, y = self._sample_brick_xy()
        self.brick.set_world_pose(position=np.array([x, y, self.table_height + 0.01], dtype=np.float32))
        self.brick.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.brick.set_angular_velocity(np.zeros(3, dtype=np.float32))
        self._step_world(steps=10)

        observation = self._get_observation()
        return self._flatten_obs(observation), {}

    def step(self, action: np.ndarray):
        self.current_step += 1
        current_observation = self._get_observation()
        arm_dim = len(self.training_config.arm_joints)

        arm_delta = action[:arm_dim] * 0.08
        arm_targets = np.clip(current_observation["arm_positions"] + arm_delta, self.arm_lower, self.arm_upper)
        gripper_action = action[arm_dim:]
        gripper_targets = np.clip((gripper_action + 1.0) * 0.015, 0.0, 0.03)

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
        if terminated or truncated:
            info["episode_completed"] = True
            info["episode_reward"] = self._task_metrics.cumulative_reward
            info["episode_length"] = float(self.current_step)
        return self._flatten_obs(new_observation), reward, terminated, truncated, info

    def close(self) -> None:
        return None
