from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.build_mjcf import SCENE_XML_PATH, ensure_assets_built


class PickBrickEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(
        self,
        render_mode: str | None = None,
        max_episode_steps: int = 250,
        frame_skip: int = 10,
    ) -> None:
        ensure_assets_built()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.frame_skip = frame_skip
        self.delta_scale = np.array([0.05, 0.05, 0.05, 0.06, 0.08, 0.10], dtype=np.float32)
        self.table_top_z = 0.75
        self.brick_half_height = 0.01
        self.success_height = 0.87
        self.workspace_low = np.array([0.10, 0.05, 0.60], dtype=np.float64)
        self.workspace_high = np.array([0.48, 0.45, 1.15], dtype=np.float64)
        self.required_success_steps = 8

        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.viewer = None

        self.arm_joint_names = [
            "right_base_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]
        self.finger_joint_names = ["right_gripper1_joint", "right_gripper2_joint"]
        self.all_joint_names = self.arm_joint_names + self.finger_joint_names
        self.actuator_names = [f"{name}_act" for name in self.all_joint_names]
        self.home_qpos = {
            "right_base_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            "right_gripper1_joint": 0.025,
            "right_gripper2_joint": -0.025,
        }

        self.joint_ids = {name: self._name2id(mujoco.mjtObj.mjOBJ_JOINT, name) for name in self.all_joint_names}
        self.actuator_ids = {name: self._name2id(mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in self.actuator_names}
        self.site_ids = {
            "ee_site": self._name2id(mujoco.mjtObj.mjOBJ_SITE, "ee_site"),
            "brick_site": self._name2id(mujoco.mjtObj.mjOBJ_SITE, "brick_site"),
        }
        self.geom_ids = {
            "brick_geom": self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "brick_geom"),
            "finger_left_pad": self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "finger_left_pad"),
            "finger_right_pad": self._name2id(mujoco.mjtObj.mjOBJ_GEOM, "finger_right_pad"),
        }
        self.brick_joint_id = self._name2id(mujoco.mjtObj.mjOBJ_JOINT, "brick_freejoint")
        self.brick_qpos_adr = self.model.jnt_qposadr[self.brick_joint_id]
        self.brick_qvel_adr = self.model.jnt_dofadr[self.brick_joint_id]
        self.ee_site_id = self.site_ids["ee_site"]
        self.brick_site_id = self.site_ids["brick_site"]

        arm_qpos_dim = len(self.arm_joint_names)
        arm_qvel_dim = len(self.arm_joint_names)
        obs_dim = arm_qpos_dim + arm_qvel_dim + 1 + 3 + 3 + 3 + 1
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        self._last_action = np.zeros(7, dtype=np.float32)
        self._ctrl_targets = np.zeros(len(self.all_joint_names), dtype=np.float64)
        self._success_counter = 0
        self._elapsed_steps = 0
        self._seed = None

    def _name2id(self, obj_type: mujoco.mjtObj, name: str) -> int:
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)
        if obj_id < 0:
            raise KeyError(f"Missing MuJoCo object: {name}")
        return obj_id

    def _joint_qpos(self, name: str) -> float:
        joint_id = self.joint_ids[name]
        return float(self.data.qpos[self.model.jnt_qposadr[joint_id]])

    def _joint_qvel(self, name: str) -> float:
        joint_id = self.joint_ids[name]
        return float(self.data.qvel[self.model.jnt_dofadr[joint_id]])

    def _set_joint_qpos(self, name: str, value: float) -> None:
        joint_id = self.joint_ids[name]
        self.data.qpos[self.model.jnt_qposadr[joint_id]] = value

    def _set_joint_qvel(self, name: str, value: float) -> None:
        joint_id = self.joint_ids[name]
        self.data.qvel[self.model.jnt_dofadr[joint_id]] = value

    def _joint_range(self, name: str) -> tuple[float, float]:
        joint_id = self.joint_ids[name]
        low, high = self.model.jnt_range[joint_id]
        return float(low), float(high)

    def _brick_position(self) -> np.ndarray:
        return self.data.site_xpos[self.brick_site_id].copy()

    def _ee_position(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def _gripper_opening(self) -> float:
        return self._joint_qpos("right_gripper1_joint") - self._joint_qpos("right_gripper2_joint")

    def _detect_grasp(self) -> bool:
        left_touch = False
        right_touch = False
        brick_geom = self.geom_ids["brick_geom"]
        left_geom = self.geom_ids["finger_left_pad"]
        right_geom = self.geom_ids["finger_right_pad"]

        for idx in range(self.data.ncon):
            contact = self.data.contact[idx]
            geom_a = int(contact.geom1)
            geom_b = int(contact.geom2)
            pair = {geom_a, geom_b}
            if brick_geom not in pair:
                continue
            if left_geom in pair:
                left_touch = True
            if right_geom in pair:
                right_touch = True
        brick_height = self._brick_position()[2]
        return left_touch and right_touch and brick_height > self.table_top_z + self.brick_half_height - 0.002

    def _compute_observation(self) -> np.ndarray:
        joint_positions = np.array([self._joint_qpos(name) for name in self.arm_joint_names], dtype=np.float32)
        joint_velocities = np.array([self._joint_qvel(name) for name in self.arm_joint_names], dtype=np.float32)
        gripper = np.array([self._gripper_opening()], dtype=np.float32)
        ee_pos = self._ee_position().astype(np.float32)
        brick_pos = self._brick_position().astype(np.float32)
        rel = (brick_pos - ee_pos).astype(np.float32)
        brick_height = np.array([brick_pos[2]], dtype=np.float32)
        return np.concatenate([joint_positions, joint_velocities, gripper, ee_pos, brick_pos, rel, brick_height])

    def _set_ctrl_targets_from_state(self) -> None:
        for idx, name in enumerate(self.all_joint_names):
            self._ctrl_targets[idx] = self._joint_qpos(name)
            actuator_id = self.actuator_ids[f"{name}_act"]
            self.data.ctrl[actuator_id] = self._ctrl_targets[idx]

    def _sample_brick_position(self) -> np.ndarray:
        xy = self.np_random.uniform(low=np.array([0.23, 0.22]), high=np.array([0.33, 0.32]))
        return np.array([xy[0], xy[1], self.table_top_z + self.brick_half_height], dtype=np.float64)

    def _reset_sim_state(self) -> None:
        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.act[:] = 0.0

        for name, value in self.home_qpos.items():
            self._set_joint_qpos(name, value)
            self._set_joint_qvel(name, 0.0)

        brick_pos = self._sample_brick_position()
        brick_qpos = np.array([brick_pos[0], brick_pos[1], brick_pos[2], 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[self.brick_qpos_adr : self.brick_qpos_adr + 7] = brick_qpos
        self.data.qvel[self.brick_qvel_adr : self.brick_qvel_adr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)
        self._set_ctrl_targets_from_state()
        self._success_counter = 0
        self._elapsed_steps = 0
        self._last_action[:] = 0.0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._seed = seed
        self._reset_sim_state()
        observation = self._compute_observation()
        info = {
            "distance_to_brick": float(np.linalg.norm(self._brick_position() - self._ee_position())),
            "brick_height": float(self._brick_position()[2]),
            "grasp_detected": False,
            "success": False,
            "seed": self._seed,
        }
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self._elapsed_steps += 1

        for index, joint_name in enumerate(self.arm_joint_names):
            low, high = self._joint_range(joint_name)
            self._ctrl_targets[index] = np.clip(
                self._ctrl_targets[index] + float(action[index]) * float(self.delta_scale[index]),
                low,
                high,
            )

        open_fraction = 0.5 * (float(action[6]) + 1.0)
        finger_1_target = 0.028 * open_fraction
        finger_2_target = -finger_1_target
        self._ctrl_targets[6] = finger_1_target
        self._ctrl_targets[7] = finger_2_target

        for idx, joint_name in enumerate(self.all_joint_names):
            actuator_id = self.actuator_ids[f"{joint_name}_act"]
            self.data.ctrl[actuator_id] = self._ctrl_targets[idx]

        mujoco.mj_step(self.model, self.data, nstep=self.frame_skip)

        ee_pos = self._ee_position()
        brick_pos = self._brick_position()
        distance = float(np.linalg.norm(ee_pos - brick_pos))
        brick_height = float(brick_pos[2])
        grasp_detected = self._detect_grasp()
        useful_lift = max(0.0, brick_height - (self.table_top_z + self.brick_half_height))
        action_penalty = 0.01 * float(np.dot(action, action))
        smoothness_penalty = 0.005 * float(np.dot(action - self._last_action, action - self._last_action))

        reach_reward = -distance
        grasp_bonus = 0.5 if grasp_detected else 0.0
        lift_reward = 6.0 * useful_lift
        success_height_bonus = 2.0 if brick_height > self.success_height else 0.0
        reward = reach_reward + grasp_bonus + lift_reward + success_height_bonus - action_penalty - smoothness_penalty

        success_now = grasp_detected and brick_height > self.success_height
        if success_now:
            self._success_counter += 1
        else:
            self._success_counter = 0

        success = self._success_counter >= self.required_success_steps
        if success:
            reward += 25.0

        out_of_workspace = bool(np.any(brick_pos < self.workspace_low) or np.any(brick_pos > self.workspace_high))
        brick_fell = bool(brick_height < self.table_top_z - 0.12)
        terminated = success or out_of_workspace or brick_fell
        truncated = self._elapsed_steps >= self.max_episode_steps

        self._last_action = action.copy()
        observation = self._compute_observation()
        info = {
            "distance_to_brick": distance,
            "brick_height": brick_height,
            "grasp_detected": grasp_detected,
            "success": success,
            "reward_reach": reach_reward,
            "reward_grasp": grasp_bonus,
            "reward_lift": lift_reward,
            "reward_success_height": success_height_bonus,
            "reward_action_penalty": -action_penalty,
            "reward_smoothness_penalty": -smoothness_penalty,
            "out_of_workspace": out_of_workspace,
            "brick_fell": brick_fell,
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, 640, 480)
            self.renderer.update_scene(self.data, camera="overview")
            return self.renderer.render()

        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer

                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None

        return None

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
