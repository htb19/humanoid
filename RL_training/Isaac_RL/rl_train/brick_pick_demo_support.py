from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .config import RobotTrainingConfig
from .isaac_import import import_urdf, verify_runtime_urdf
from .pose_math import Pose, quat_wxyz_to_matrix


_SIMULATION_APP = None


def get_simulation_app(headless: bool):
    global _SIMULATION_APP
    if _SIMULATION_APP is not None:
        return _SIMULATION_APP

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    _SIMULATION_APP = SimulationApp({"headless": headless})
    return _SIMULATION_APP


def load_isaac_modules() -> dict[str, object]:
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
        from isaacsim.core.prims import SingleArticulation
        from isaacsim.core.utils.prims import get_prim_at_path
        from isaacsim.core.utils.types import ArticulationAction
    except ImportError:
        from omni.isaac.core import World
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
        from omni.isaac.core.articulations import Articulation as SingleArticulation
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.types import ArticulationAction

    from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
    from pxr import UsdGeom

    return {
        "World": World,
        "DynamicCuboid": DynamicCuboid,
        "FixedCuboid": FixedCuboid,
        "SingleArticulation": SingleArticulation,
        "get_prim_at_path": get_prim_at_path,
        "ArticulationAction": ArticulationAction,
        "LulaKinematicsSolver": LulaKinematicsSolver,
        "UsdGeom": UsdGeom,
    }


def _parse_actuated_joint_names(urdf_path: Path) -> list[str]:
    root = ET.parse(urdf_path).getroot()
    actuated = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type", "fixed") in {"fixed", "floating", "planar"}:
            continue
        actuated.append(joint.attrib["name"])
    return actuated


def _parse_urdf_link_names(urdf_path: Path) -> set[str]:
    root = ET.parse(urdf_path).getroot()
    return {link.attrib["name"] for link in root.findall("link")}


def parse_joint_origins(urdf_path: Path) -> dict[str, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    joint_origins: dict[str, np.ndarray] = {}
    for joint in root.findall("joint"):
        origin = joint.find("origin")
        xyz = np.zeros(3, dtype=np.float64)
        if origin is not None:
            xyz = np.fromstring(origin.attrib.get("xyz", "0 0 0"), sep=" ", dtype=np.float64)
        joint_origins[joint.attrib["name"]] = xyz
    return joint_origins


def validate_lula_inputs(training_config: RobotTrainingConfig) -> None:
    link_names = _parse_urdf_link_names(training_config.urdf_path)
    chosen_root_link = training_config.urdf_root_link
    chosen_ee_link = training_config.end_effector_link

    print(
        "[demo] lula config validation",
        {
            "root_link": chosen_root_link,
            "end_effector_link": chosen_ee_link,
            "urdf_path": str(training_config.urdf_path),
        },
    )

    if chosen_root_link not in link_names:
        raise ValueError(
            f"Lula root link '{chosen_root_link}' does not exist in URDF {training_config.urdf_path}."
        )
    if chosen_ee_link not in link_names:
        raise ValueError(
            f"Lula end-effector link '{chosen_ee_link}' does not exist in URDF {training_config.urdf_path}."
        )


def materialize_lula_robot_description(training_config: RobotTrainingConfig) -> Path:
    runtime_dir = Path(tempfile.gettempdir()) / "rl_train"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    descriptor_path = runtime_dir / "humanoid_right_arm_lula_descriptor.yaml"

    all_actuated = _parse_actuated_joint_names(training_config.urdf_path)
    fixed_rules = []
    for joint_name in all_actuated:
        if joint_name in training_config.arm_joints:
            continue
        fixed_rules.append(f"  - {{name: {joint_name}, rule: fixed, value: 0.0}}")

    default_q = ", ".join("0.0" for _ in training_config.arm_joints)
    accel_limits = ", ".join("20.0" for _ in training_config.arm_joints)
    jerk_limits = ", ".join("200.0" for _ in training_config.arm_joints)
    lines = [
        "api_version: 1.0",
        "",
        "cspace:",
        *[f"  - {joint_name}" for joint_name in training_config.arm_joints],
        "",
        f"root_link: {training_config.urdf_root_link}",
        "",
        f"default_q: [{default_q}]",
        f"acceleration_limits: [{accel_limits}]",
        f"jerk_limits: [{jerk_limits}]",
        "",
    ]
    if fixed_rules:
        lines.extend(["cspace_to_urdf_rules:", *fixed_rules])
    else:
        lines.append("cspace_to_urdf_rules: []")
    lines.extend(["", "collision_spheres: []"])

    descriptor_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return descriptor_path


@dataclass(frozen=True)
class DemoSceneConfig:
    physics_dt: float = 1.0 / 60.0
    control_dt: float = 1.0 / 30.0
    table_position: tuple[float, float, float] = (0.14, 0.64, 0.28)
    table_scale: tuple[float, float, float] = (0.72, 0.56, 0.06)
    brick_position: tuple[float, float, float] = (0.16, 0.56, 0.355)
    brick_scale: tuple[float, float, float] = (0.08, 0.04, 0.05)
    place_position: tuple[float, float, float] = (0.04, 0.56, 0.355)

    @property
    def table_height(self) -> float:
        return self.table_position[2] + (self.table_scale[2] / 2.0)


class HumanoidBrickPickDemoScene:
    def __init__(self, training_config: RobotTrainingConfig, headless: bool) -> None:
        self.training_config = training_config
        self.headless = headless
        self.scene_config = DemoSceneConfig()
        self._app = get_simulation_app(headless)
        self._modules = load_isaac_modules()

        self._World = self._modules["World"]
        self._DynamicCuboid = self._modules["DynamicCuboid"]
        self._FixedCuboid = self._modules["FixedCuboid"]
        self._SingleArticulation = self._modules["SingleArticulation"]
        self._get_prim_at_path = self._modules["get_prim_at_path"]
        self._ArticulationAction = self._modules["ArticulationAction"]
        self._LulaKinematicsSolver = self._modules["LulaKinematicsSolver"]
        self._UsdGeom = self._modules["UsdGeom"]

        self.world = self._World(
            stage_units_in_meters=1.0,
            physics_dt=self.scene_config.physics_dt,
            rendering_dt=self.scene_config.physics_dt,
        )
        self.world.scene.add_default_ground_plane()

        self.table = self.world.scene.add(
            self._FixedCuboid(
                prim_path="/World/demo_table",
                name="demo_table",
                position=np.array(self.scene_config.table_position, dtype=np.float32),
                scale=np.array(self.scene_config.table_scale, dtype=np.float32),
                color=np.array([0.48, 0.32, 0.18], dtype=np.float32),
            )
        )
        self.brick = self.world.scene.add(
            self._DynamicCuboid(
                prim_path="/World/demo_brick",
                name="demo_brick",
                position=np.array(self.scene_config.brick_position, dtype=np.float32),
                scale=np.array(self.scene_config.brick_scale, dtype=np.float32),
                color=np.array([0.85, 0.18, 0.12], dtype=np.float32),
                mass=0.05,
            )
        )

        print("[demo] source URDF:", self.training_config.urdf_path)
        verify_runtime_urdf(self.training_config.runtime_urdf_path)
        self.robot_prim_path = import_urdf(self.training_config.runtime_urdf_path)

        self.articulation = self.world.scene.add(
            self._SingleArticulation(prim_path=self.robot_prim_path, name="humanoid")
        )
        self.world.reset()
        self.articulation.initialize()

        self.dof_names = list(self.articulation.dof_names)
        self.arm_indices = [self.dof_names.index(name) for name in self.training_config.arm_joints]
        self.gripper_indices = [self.dof_names.index(name) for name in self.training_config.gripper_joints]
        self.arm_lower, self.arm_upper = self._build_arm_bounds()

        self.base_prim = self._get_prim_at_path(f"{self.robot_prim_path}/base_link")
        self.ee_prim = self._get_prim_at_path(f"{self.robot_prim_path}/{self.training_config.end_effector_link}")

        validate_lula_inputs(self.training_config)
        self.robot_descriptor_path = materialize_lula_robot_description(training_config)
        self.kinematics = self._LulaKinematicsSolver(
            robot_description_path=str(self.robot_descriptor_path),
            urdf_path=str(self.training_config.runtime_urdf_path),
        )
        self.kinematics.set_default_position_tolerance(0.005)
        self.kinematics.set_default_orientation_tolerance(0.08)
        self.kinematics.set_default_cspace_seeds(
            np.array([np.zeros(len(self.training_config.arm_joints), dtype=np.float64)], dtype=np.float64)
        )

    def _build_arm_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower = []
        upper = []
        debug_bounds = {}
        for joint_name in self.training_config.arm_joints:
            limit = self.training_config.joint_limits.get(joint_name, {})
            low = float(limit.get("min_position", -3.14))
            high = float(limit.get("max_position", 3.14))
            lower.append(low)
            upper.append(high)
            debug_bounds[joint_name] = {
                "min": round(low, 4),
                "max": round(high, 4),
                "source": limit.get("limit_source", "urdf"),
            }
        print("[demo] arm joint bounds", debug_bounds)
        return np.array(lower, dtype=np.float64), np.array(upper, dtype=np.float64)

    def get_link_pose(self, link_name: str) -> Pose:
        prim = self._get_prim_at_path(f"{self.robot_prim_path}/{link_name}")
        return self.get_prim_pose(prim)

    def get_prim_pose(self, prim) -> Pose:
        transform = self._UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0.0)
        translation = transform.ExtractTranslation()
        rotation = np.array(transform.ExtractRotationMatrix(), dtype=np.float64)
        return Pose(
            position=np.array([float(translation[0]), float(translation[1]), float(translation[2])], dtype=np.float64),
            rotation=rotation,
        )

    def get_base_pose(self) -> Pose:
        return self.get_prim_pose(self.base_prim)

    def get_end_effector_pose(self) -> Pose:
        return self.get_prim_pose(self.ee_prim)

    def get_brick_pose(self) -> Pose:
        position, orientation = self.brick.get_world_pose()
        return Pose(position=np.array(position, dtype=np.float64), rotation=quat_wxyz_to_matrix(np.array(orientation)))

    def current_arm_positions(self) -> np.ndarray:
        positions = np.array(self.articulation.get_joint_positions(), dtype=np.float64)
        return positions[self.arm_indices]

    def current_gripper_positions(self) -> np.ndarray:
        positions = np.array(self.articulation.get_joint_positions(), dtype=np.float64)
        return positions[self.gripper_indices]

    def apply_joint_targets(self, arm_target: np.ndarray, gripper_target: np.ndarray) -> None:
        full_target = np.array(self.articulation.get_joint_positions(), dtype=np.float64)
        full_target[self.arm_indices] = np.clip(np.array(arm_target, dtype=np.float64), self.arm_lower, self.arm_upper)
        full_target[self.gripper_indices] = np.array(gripper_target, dtype=np.float64)
        self.articulation.apply_action(self._ArticulationAction(joint_positions=full_target))

    def set_robot_home(self) -> None:
        full_positions = np.zeros(len(self.dof_names), dtype=np.float64)
        self.articulation.set_joint_positions(full_positions)
        self.articulation.set_joint_velocities(np.zeros(len(self.dof_names), dtype=np.float64))
        self.articulation.apply_action(self._ArticulationAction(joint_positions=full_positions))

    def reset_scene(self) -> None:
        self.world.reset()
        self.articulation.initialize()
        self.set_robot_home()
        self.brick.set_world_pose(position=np.array(self.scene_config.brick_position, dtype=np.float32))
        self.brick.set_linear_velocity(np.zeros(3, dtype=np.float32))
        self.brick.set_angular_velocity(np.zeros(3, dtype=np.float32))
        self.step_world(steps=15)

    def step_world(self, steps: int = 1) -> None:
        for _ in range(max(1, steps)):
            self.world.step(render=not self.headless)

    def sync_kinematics_base_pose(self) -> None:
        base_pose = self.get_base_pose()
        self.kinematics.set_robot_base_pose(base_pose.position, base_pose.quaternion_wxyz)

    def solve_ik(
        self,
        target_pose: Pose,
        warm_start: np.ndarray | None = None,
        extra_seed: np.ndarray | None = None,
        position_tolerance: float = 0.005,
        orientation_tolerance: float = 0.1,
    ) -> tuple[np.ndarray | None, bool]:
        self.sync_kinematics_base_pose()
        if warm_start is None:
            warm_start = self.current_arm_positions()
        seeds = [np.array(warm_start, dtype=np.float64), np.zeros(len(self.training_config.arm_joints), dtype=np.float64)]
        if extra_seed is not None:
            seeds.insert(1, np.array(extra_seed, dtype=np.float64))
        self.kinematics.set_default_cspace_seeds(
            np.array(seeds, dtype=np.float64)
        )
        result, success = self.kinematics.compute_inverse_kinematics(
            self.training_config.end_effector_link,
            target_pose.position,
            target_pose.quaternion_wxyz,
            warm_start=np.array(warm_start, dtype=np.float64),
            position_tolerance=position_tolerance,
            orientation_tolerance=orientation_tolerance,
        )
        if not success:
            return None, False
        return np.array(result, dtype=np.float64), True

    def close(self) -> None:
        return None
