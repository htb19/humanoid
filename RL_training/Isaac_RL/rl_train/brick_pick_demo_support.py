from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .config import RobotTrainingConfig
from .isaac_import import import_urdf, verify_runtime_urdf
from .pose_math import Pose, matrix_to_quat_wxyz, quat_wxyz_to_matrix


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
        fixed_value = training_config.home_joint_positions.get(
            joint_name, training_config.closed_gripper_positions.get(joint_name, 0.0)
        )
        fixed_rules.append(f"  - {{name: {joint_name}, rule: fixed, value: {fixed_value:.6f}}}")

    default_q = ", ".join(
        f"{training_config.home_joint_positions.get(joint_name, 0.0):.6f}"
        for joint_name in training_config.arm_joints
    )
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
    table_position: tuple[float, float, float] = (0.48, -0.24, 0.28)
    table_scale: tuple[float, float, float] = (0.70, 0.50, 0.06)
    brick_position: tuple[float, float, float] = (0.42, -0.28, 0.355)
    brick_scale: tuple[float, float, float] = (0.08, 0.04, 0.05)
    place_position: tuple[float, float, float] = (0.42, -0.14, 0.355)

    @property
    def table_height(self) -> float:
        return self.table_position[2] + (self.table_scale[2] / 2.0)


@dataclass(frozen=True)
class MountedCameraSpec:
    camera_name: str
    mount_link: str
    translation_xyz: tuple[float, float, float]
    rotation_xyz_deg: tuple[float, float, float]
    validation_joint_name: str


class HumanoidBrickPickDemoScene:
    def __init__(self, training_config: RobotTrainingConfig, headless: bool, setup_cameras: bool = True) -> None:
        self.training_config = training_config
        self.headless = headless
        self.setup_cameras = setup_cameras
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
        self.camera_mount_specs = {
            "head_camera": MountedCameraSpec(
                camera_name="head_camera",
                mount_link="head_camera_link",
                translation_xyz=(0.05, 0.0, 0.05),
                rotation_xyz_deg=(0.0, -25.0, 0.0),
                validation_joint_name="neck_yaw_joint",
            ),
            "left_arm_camera": MountedCameraSpec(
                camera_name="left_arm_camera",
                mount_link="left_camera_link",
                translation_xyz=(0.045, 0.0, 0.025),
                rotation_xyz_deg=(0.0, -25.0, 0.0),
                validation_joint_name="left_wrist_yaw_joint",
            ),
            "right_arm_camera": MountedCameraSpec(
                camera_name="right_arm_camera",
                mount_link="right_camera_link",
                translation_xyz=(0.045, 0.0, 0.025),
                rotation_xyz_deg=(0.0, -25.0, 0.0),
                validation_joint_name="right_wrist_yaw_joint",
            ),
        }
        self.camera_prim_paths: dict[str, str | None] = {
            camera_name: None for camera_name in self.camera_mount_specs
        }
        self._camera_viewport_windows: list[object] = []

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
        self.robot_prim = self._get_prim_at_path(self.robot_prim_path)

        self.articulation = self.world.scene.add(
            self._SingleArticulation(prim_path=self.robot_prim_path, name="humanoid")
        )
        self.world.reset()
        self.articulation.initialize()
        self.set_robot_root_pose()

        self.dof_names = list(self.articulation.dof_names)
        self.arm_indices = [self.dof_names.index(name) for name in self.training_config.arm_joints]
        self.gripper_indices = [self.dof_names.index(name) for name in self.training_config.gripper_joints]
        self.arm_lower, self.arm_upper = self._build_arm_bounds()
        if self.setup_cameras:
            self.setup_onboard_cameras()

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
            np.array([self.home_arm_positions()], dtype=np.float64)
        )

    def _camera_prim_path(self, spec: MountedCameraSpec) -> str:
        return f"{self.robot_prim_path}/{spec.mount_link}/{spec.camera_name}"

    def _camera_mount_parent_path(self, spec: MountedCameraSpec) -> str:
        return f"{self.robot_prim_path}/{spec.mount_link}"

    def _camera_rotation_matrix(self, rotation_xyz_deg: tuple[float, float, float]) -> np.ndarray:
        rx, ry, rz = [math.radians(value) for value in rotation_xyz_deg]
        cx, sx = math.cos(rx), math.sin(rx)
        cy, sy = math.cos(ry), math.sin(ry)
        cz, sz = math.cos(rz), math.sin(rz)
        rot_x = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]], dtype=np.float64)
        rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
        rot_z = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        return rot_z @ rot_y @ rot_x

    def _is_arm_camera(self, spec: MountedCameraSpec) -> bool:
        return spec.camera_name in {"left_arm_camera", "right_arm_camera"}

    def _arm_target_link(self, spec: MountedCameraSpec) -> str:
        if spec.camera_name == "left_arm_camera":
            return "left_wrist_yaw_link"
        return "right_wrist_yaw_link"

    def _look_at_rotation(self, camera_position: np.ndarray, target_position: np.ndarray) -> np.ndarray:
        forward = np.array(target_position - camera_position, dtype=np.float64)
        forward /= max(np.linalg.norm(forward), 1e-9)
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        lateral = np.cross(world_up, forward)
        if np.linalg.norm(lateral) < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            lateral = np.cross(world_up, forward)
        lateral /= max(np.linalg.norm(lateral), 1e-9)
        vertical = np.cross(forward, lateral)
        vertical /= max(np.linalg.norm(vertical), 1e-9)
        return np.column_stack([forward, lateral, vertical])

    def _camera_local_transform(self, spec: MountedCameraSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        local_translation = np.array(spec.translation_xyz, dtype=np.float64)
        if not self._is_arm_camera(spec):
            return local_translation, self._camera_rotation_matrix(spec.rotation_xyz_deg), None

        mount_link_pose = self.get_link_pose(spec.mount_link)
        target_link_pose = self.get_link_pose(self._arm_target_link(spec))
        camera_world_position = mount_link_pose.position + mount_link_pose.rotation @ local_translation
        approach_direction = np.array(target_link_pose.rotation[:, 0], dtype=np.float64)
        approach_direction /= max(np.linalg.norm(approach_direction), 1e-9)
        look_at_target = target_link_pose.position + (0.12 * approach_direction) + np.array([0.0, 0.0, -0.07], dtype=np.float64)
        world_rotation = self._look_at_rotation(camera_world_position, look_at_target)
        local_rotation = mount_link_pose.rotation.T @ world_rotation
        return local_translation, local_rotation, look_at_target

    def _find_named_camera_prims(self, camera_name: str) -> list[object]:
        matches = []
        stage = self.world.stage
        for prim in stage.Traverse():
            if not prim.IsValid() or not prim.IsA(self._UsdGeom.Camera):
                continue
            path_string = prim.GetPath().pathString
            if path_string.startswith(self.robot_prim_path) and prim.GetName() == camera_name:
                matches.append(prim)
        return matches

    def _inspect_camera_candidates(self) -> None:
        for camera_name, spec in self.camera_mount_specs.items():
            matches = self._find_named_camera_prims(camera_name)
            if not matches:
                print(f"[demo] inspect {camera_name}: no existing camera prims found before mounting")
                continue
            for prim in matches:
                parent_path = prim.GetParent().GetPath().pathString
                expected_parent = self._camera_mount_parent_path(spec)
                status = "OK" if parent_path == expected_parent else "MISMATCH"
                print(
                    "[demo] inspect camera",
                    {
                        "camera_name": camera_name,
                        "prim_path": prim.GetPath().pathString,
                        "parent_path": parent_path,
                        "expected_parent": expected_parent,
                        "status": status,
                    },
                )

    def _ensure_mounted_camera(self, spec: MountedCameraSpec) -> str | None:
        from pxr import Gf

        mount_parent_path = self._camera_mount_parent_path(spec)
        mount_parent = self._get_prim_at_path(mount_parent_path)
        if not mount_parent or not mount_parent.IsValid():
            print(
                f"[warning] {spec.camera_name} could not be mounted because link prim is missing: {mount_parent_path}"
            )
            return None

        camera_path = self._camera_prim_path(spec)
        camera_prim = self._UsdGeom.Camera.Define(self.world.stage, camera_path)
        xformable = self._UsdGeom.Xformable(camera_prim)
        local_translation, local_rotation, _ = self._camera_local_transform(spec)
        local_quaternion = matrix_to_quat_wxyz(local_rotation)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*local_translation.tolist()))
        xformable.AddOrientOp().Set(
            Gf.Quatf(
                float(local_quaternion[0]),
                float(local_quaternion[1]),
                float(local_quaternion[2]),
                float(local_quaternion[3]),
            )
        )
        camera_prim.CreateProjectionAttr("perspective")
        camera_prim.CreateFocalLengthAttr(18.0)
        camera_prim.CreateHorizontalApertureAttr(20.955)
        camera_prim.CreateVerticalApertureAttr(15.2908)
        camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        return camera_path

    def _camera_local_pose(self, spec: MountedCameraSpec) -> tuple[Pose, np.ndarray | None]:
        local_translation, local_rotation, look_at_target = self._camera_local_transform(spec)
        return Pose(position=local_translation, rotation=local_rotation), look_at_target

    def _print_camera_mount_report(self, camera_name: str, spec: MountedCameraSpec, camera_path: str | None) -> None:
        if camera_path is None:
            print(f"[warning] {camera_name} prim path unresolved")
            return

        camera_prim = self._get_prim_at_path(camera_path)
        if not camera_prim or not camera_prim.IsValid():
            print(f"[warning] {camera_name} resolved to an invalid prim: {camera_path}")
            return

        parent_path = camera_prim.GetParent().GetPath().pathString
        local_pose, look_at_target = self._camera_local_pose(spec)
        world_pose = self.get_prim_pose(camera_prim)
        world_forward = np.array(world_pose.rotation[:, 0], dtype=np.float64)
        report = {
            "camera_name": camera_name,
            "prim_path": camera_path,
            "parent_path": parent_path,
            "mount_link": spec.mount_link,
            "local_translation": np.round(local_pose.position, 4).tolist(),
            "local_rotation_quaternion_wxyz": np.round(matrix_to_quat_wxyz(local_pose.rotation), 4).tolist(),
            "world_position": np.round(world_pose.position, 4).tolist(),
            "world_quaternion_wxyz": np.round(world_pose.quaternion_wxyz, 4).tolist(),
            "world_forward_direction": np.round(world_forward, 4).tolist(),
        }
        if look_at_target is not None:
            report["look_at_target"] = np.round(look_at_target, 4).tolist()
        print("[demo] mounted camera", report)

        if self._is_arm_camera(spec):
            if abs(float(world_forward[2])) > 0.92:
                print(f"[warning] {camera_name} optical axis is too close to vertical")
            base_prim = self._get_prim_at_path(f"{self.robot_prim_path}/base_link")
            if base_prim and base_prim.IsValid():
                base_pose = self.get_prim_pose(base_prim)
                to_robot_body = base_pose.position - world_pose.position
                to_robot_body /= max(np.linalg.norm(to_robot_body), 1e-9)
                if float(np.dot(world_forward, to_robot_body)) > 0.25:
                    print(f"[warning] {camera_name} optical axis points back toward the robot body")

    def _validate_camera_attachment(self, camera_name: str, spec: MountedCameraSpec, camera_path: str | None) -> None:
        if camera_path is None:
            return
        if spec.validation_joint_name not in self.dof_names:
            print(f"[warning] could not validate {camera_name}; missing joint {spec.validation_joint_name}")
            return

        camera_prim = self._get_prim_at_path(camera_path)
        if not camera_prim or not camera_prim.IsValid():
            print(f"[warning] could not validate {camera_name}; invalid prim {camera_path}")
            return

        joint_index = self.dof_names.index(spec.validation_joint_name)
        original_positions = np.array(self.articulation.get_joint_positions(), dtype=np.float64)
        before_pose = self.get_prim_pose(camera_prim)
        moved_positions = np.array(original_positions, dtype=np.float64)
        moved_positions[joint_index] += 0.08

        self.articulation.set_joint_positions(moved_positions)
        self.articulation.set_joint_velocities(np.zeros(len(self.dof_names), dtype=np.float64))
        self.step_world(steps=4)
        after_pose = self.get_prim_pose(camera_prim)

        self.articulation.set_joint_positions(original_positions)
        self.articulation.set_joint_velocities(np.zeros(len(self.dof_names), dtype=np.float64))
        self.step_world(steps=4)

        translation_delta = float(np.linalg.norm(after_pose.position - before_pose.position))
        rotation_delta = float(np.linalg.norm(after_pose.quaternion_wxyz - before_pose.quaternion_wxyz))
        before_link_pose = self.get_link_pose(spec.mount_link)
        link_offset_before = np.array(before_pose.position - before_link_pose.position, dtype=np.float64)
        if np.linalg.norm(link_offset_before) < 0.01:
            print(f"[warning] {camera_name} may still be inside link geometry")
        print(
            "[demo] camera attachment validation",
            {
                "camera_name": camera_name,
                "joint_name": spec.validation_joint_name,
                "translation_delta": round(translation_delta, 6),
                "rotation_delta": round(rotation_delta, 6),
                "offset_from_link_origin": np.round(link_offset_before, 4).tolist(),
            },
        )
        if translation_delta < 1e-5 and rotation_delta < 1e-5:
            print("Camera is not actually attached to robot link")

    def _setup_camera_viewports(self) -> None:
        if self.headless:
            return

        try:
            from omni.kit.viewport.utility import create_viewport_window, get_active_viewport_window
        except Exception as exc:
            print(f"[warning] viewport utility import failed; skipping onboard camera GUI setup: {exc}")
            return

        main_viewport_window = get_active_viewport_window()
        if main_viewport_window is None:
            print("[warning] main perspective viewport was not found; skipping onboard camera GUI setup")
            return

        viewport_specs = [
            ("Head Camera", "head_camera", 20, 120, 480, 270),
            ("Left Arm Camera", "left_arm_camera", 520, 120, 480, 270),
            ("Right Arm Camera", "right_arm_camera", 1020, 120, 480, 270),
        ]
        self._camera_viewport_windows = []

        for window_name, camera_name, pos_x, pos_y, width, height in viewport_specs:
            camera_path = self.camera_prim_paths.get(camera_name)
            if camera_path is None:
                print(f"[warning] skipping viewport '{window_name}' because {camera_name} is unavailable")
                continue

            viewport_window = create_viewport_window(
                name=window_name,
                width=width,
                height=height,
                position_x=pos_x,
                position_y=pos_y,
                camera_path=camera_path,
            )
            if viewport_window is None:
                print(f"[warning] failed to create viewport '{window_name}' for camera {camera_path}")
                continue

            viewport_window.viewport_api.camera_path = camera_path
            self._camera_viewport_windows.append(viewport_window)
            print(f"[demo] viewport '{window_name}' -> {camera_path}")

    def setup_onboard_cameras(self) -> None:
        self._inspect_camera_candidates()
        self.camera_prim_paths = {}
        for camera_name, spec in self.camera_mount_specs.items():
            camera_path = self._ensure_mounted_camera(spec)
            self.camera_prim_paths[camera_name] = camera_path
            self._print_camera_mount_report(camera_name, spec, camera_path)
            self._validate_camera_attachment(camera_name, spec, camera_path)
        self._setup_camera_viewports()

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

    def set_robot_root_pose(self) -> None:
        root_position = np.array(self.training_config.robot_root_position, dtype=np.float32)
        if hasattr(self.articulation, "set_world_pose"):
            self.articulation.set_world_pose(position=root_position)
            return
        self._UsdGeom.XformCommonAPI(self.robot_prim).SetTranslate(tuple(float(v) for v in root_position))

    def home_arm_positions(self) -> np.ndarray:
        return np.array(
            [
                self.training_config.home_joint_positions.get(joint_name, 0.0)
                for joint_name in self.training_config.arm_joints
            ],
            dtype=np.float64,
        )

    def set_robot_home(self) -> None:
        full_positions = np.zeros(len(self.dof_names), dtype=np.float64)
        for joint_name, value in self.training_config.home_joint_positions.items():
            if joint_name in self.dof_names:
                full_positions[self.dof_names.index(joint_name)] = float(value)
        for joint_name, value in self.training_config.open_gripper_positions.items():
            if joint_name in self.dof_names:
                full_positions[self.dof_names.index(joint_name)] = float(value)
        self.set_robot_root_pose()
        self.articulation.set_joint_positions(full_positions)
        self.articulation.set_joint_velocities(np.zeros(len(self.dof_names), dtype=np.float64))
        self.articulation.apply_action(self._ArticulationAction(joint_positions=full_positions))

    def reset_scene(self) -> None:
        self.world.reset()
        self.articulation.initialize()
        self.set_robot_root_pose()
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
