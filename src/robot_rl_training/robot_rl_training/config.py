from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET

import yaml

try:
    from ament_index_python.packages import get_package_share_directory
except ImportError:
    get_package_share_directory = None


@dataclass(frozen=True)
class RobotTrainingConfig:
    workspace_root: Path
    robot_description_share: Path
    robot_moveit_share: Path
    urdf_path: Path
    runtime_urdf_path: Path
    srdf_path: Path
    joint_limits_path: Path
    kinematics_path: Path
    arm_group: str
    gripper_group: str
    arm_joints: list[str]
    gripper_joints: list[str]
    joint_limits: dict[str, dict]
    kinematics: dict
    end_effector_parent_link: str
    train_brick_range: dict[str, float]
    eval_brick_range: dict[str, float]


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _parse_srdf_groups(srdf_path: Path, arm_group: str, gripper_group: str) -> tuple[list[str], list[str], str]:
    root = ET.parse(srdf_path).getroot()
    groups: dict[str, list[str]] = {}
    eef_parent_link = ""

    for group in root.findall("group"):
        groups[group.attrib["name"]] = [joint.attrib["name"] for joint in group.findall("joint")]

    for eef in root.findall("end_effector"):
        if eef.attrib.get("parent_group") == arm_group and eef.attrib.get("group") == gripper_group:
            eef_parent_link = eef.attrib["parent_link"]
            break

    if arm_group not in groups:
        raise ValueError(f"Arm group '{arm_group}' not found in {srdf_path}")
    if gripper_group not in groups:
        raise ValueError(f"Gripper group '{gripper_group}' not found in {srdf_path}")
    if not eef_parent_link:
        raise ValueError(f"Could not find end effector for groups {arm_group}/{gripper_group}")

    return groups[arm_group], groups[gripper_group], eef_parent_link


def _find_workspace_root(workspace_root: Path | None = None) -> Path:
    if workspace_root is not None:
        return workspace_root.resolve()
    env_root = os.environ.get("HUMANOID_WS_ROOT")
    if env_root:
        return Path(env_root).resolve()
    return Path(__file__).resolve().parents[3]


def _find_share_from_ament_prefix(package_name: str) -> Path | None:
    prefixes = os.environ.get("AMENT_PREFIX_PATH", "").split(os.pathsep)
    for prefix in prefixes:
        if not prefix:
            continue
        share_dir = Path(prefix) / "share" / package_name
        if share_dir.exists():
            return share_dir.resolve()
    return None


def _resolve_package_share(package_name: str, workspace_root: Path) -> Path:
    if get_package_share_directory is not None:
        try:
            return Path(get_package_share_directory(package_name)).resolve()
        except Exception:
            pass

    share_dir = _find_share_from_ament_prefix(package_name)
    if share_dir is not None:
        return share_dir

    source_share = workspace_root / "src" / package_name
    if source_share.exists():
        return source_share.resolve()

    raise FileNotFoundError(f"Unable to resolve package share for '{package_name}'")


def _resolve_robot_description_share(workspace_root: Path, robot_description_path: Path | None) -> Path:
    if robot_description_path is None:
        return _resolve_package_share("robot_description", workspace_root)

    resolved = robot_description_path.resolve()
    if resolved.is_file():
        if resolved.name.endswith((".urdf", ".xacro")):
            return resolved.parent.parent.resolve()
        raise ValueError(f"Unsupported robot description file override: {resolved}")
    if (resolved / "urdf").exists():
        return resolved
    raise ValueError(
        f"--robot-description-path must point to the robot_description package root or a URDF/Xacro file: {resolved}"
    )


def _resolve_urdf_path(robot_description_share: Path, robot_description_path: Path | None) -> Path:
    if robot_description_path is not None and robot_description_path.is_file():
        return robot_description_path.resolve()

    candidates = [
        robot_description_share / "urdf" / "humanoid.urdf",
        robot_description_share / "urdf" / "humanoid.urdf.xacro",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find humanoid URDF under {robot_description_share / 'urdf'}")


def _rewrite_package_meshes(urdf_path: Path, robot_description_share: Path) -> Path:
    urdf_text = urdf_path.read_text(encoding="utf-8")
    package_root = robot_description_share.as_posix()
    urdf_text = re.sub(
        r"package://robot_description/",
        f"{package_root}/",
        urdf_text,
    )

    runtime_dir = Path(tempfile.gettempdir()) / "robot_rl_training"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_urdf_path = runtime_dir / f"{urdf_path.stem}.resolved.urdf"
    runtime_urdf_path.write_text(urdf_text, encoding="utf-8")
    return runtime_urdf_path


def load_robot_training_config(
    arm_group: str = "right_arm",
    gripper_group: str = "right_gripper",
    workspace_root: Path | None = None,
    robot_description_path: Path | None = None,
) -> RobotTrainingConfig:
    resolved_workspace_root = _find_workspace_root(workspace_root)
    robot_description_share = _resolve_robot_description_share(resolved_workspace_root, robot_description_path)
    robot_moveit_share = _resolve_package_share("robot_moveit_config", resolved_workspace_root)

    urdf_path = _resolve_urdf_path(robot_description_share, robot_description_path)
    srdf_path = robot_moveit_share / "config" / "humanoid.srdf"
    joint_limits_path = robot_moveit_share / "config" / "joint_limits.yaml"
    kinematics_path = robot_moveit_share / "config" / "kinematics.yaml"
    runtime_urdf_path = _rewrite_package_meshes(urdf_path, robot_description_share)

    arm_joints, gripper_joints, end_effector_parent_link = _parse_srdf_groups(
        srdf_path, arm_group, gripper_group
    )

    return RobotTrainingConfig(
        workspace_root=resolved_workspace_root,
        robot_description_share=robot_description_share,
        robot_moveit_share=robot_moveit_share,
        urdf_path=urdf_path,
        runtime_urdf_path=runtime_urdf_path,
        srdf_path=srdf_path,
        joint_limits_path=joint_limits_path,
        kinematics_path=kinematics_path,
        arm_group=arm_group,
        gripper_group=gripper_group,
        arm_joints=arm_joints,
        gripper_joints=gripper_joints,
        joint_limits=_load_yaml(joint_limits_path).get("joint_limits", {}),
        kinematics=_load_yaml(kinematics_path),
        end_effector_parent_link=end_effector_parent_link,
        train_brick_range={"x_min": 0.30, "x_max": 0.45, "y_min": -0.18, "y_max": 0.18},
        eval_brick_range={"x_min": 0.35, "x_max": 0.50, "y_min": -0.22, "y_max": 0.22},
    )
