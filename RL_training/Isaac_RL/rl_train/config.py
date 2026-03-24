from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import tempfile
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class RobotTrainingConfig:
    repo_root: Path
    robot_description_path: Path
    urdf_path: Path
    runtime_urdf_path: Path
    urdf_root_link: str
    arm_joints: list[str]
    gripper_joints: list[str]
    joint_limits: dict[str, dict[str, float]]
    end_effector_link: str
    train_brick_range: dict[str, float]
    eval_brick_range: dict[str, float]


def describe_file(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_file": path.is_file(),
        "readable": os.access(path, os.R_OK),
        "size_bytes": path.stat().st_size if path.exists() else None,
    }


def _resolve_repo_root(repo_root: Path | None = None) -> Path:
    if repo_root is not None:
        return repo_root.resolve()
    return Path(__file__).resolve().parents[1]


def _resolve_robot_description_path(repo_root: Path, robot_description_path: Path | None) -> Path:
    if robot_description_path is not None:
        resolved = robot_description_path.resolve()
        if not (resolved / "urdf").exists():
            raise ValueError(f"Expected robot description package root, got: {resolved}")
        return resolved

    default_path = repo_root / "assets" / "robot_description"
    if default_path.exists():
        return default_path.resolve()
    raise FileNotFoundError("Could not find assets/robot_description in the repository.")


def _resolve_urdf_path(robot_description_path: Path) -> Path:
    urdf_candidates = [
        robot_description_path / "urdf" / "humanoid.urdf",
        robot_description_path / "urdf" / "humanoid.urdf.xacro",
    ]
    for candidate in urdf_candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not find humanoid URDF under {robot_description_path / 'urdf'}")


def _rewrite_package_meshes(urdf_path: Path, robot_description_path: Path) -> Path:
    urdf_text = urdf_path.read_text(encoding="utf-8")
    urdf_text = re.sub(
        r"package://robot_description/",
        f"{robot_description_path.as_posix()}/",
        urdf_text,
    )

    runtime_dir = Path(tempfile.gettempdir()) / "rl_train"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_urdf_path = runtime_dir / f"{urdf_path.stem}.resolved.urdf"
    runtime_urdf_path.write_text(urdf_text, encoding="utf-8")
    if not runtime_urdf_path.exists() or not runtime_urdf_path.is_file():
        raise FileNotFoundError(f"Failed to materialize runtime URDF at {runtime_urdf_path}")
    if not os.access(runtime_urdf_path, os.R_OK):
        raise PermissionError(f"Runtime URDF is not readable: {runtime_urdf_path}")
    return runtime_urdf_path


def _parse_urdf_joints(
    urdf_path: Path,
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, str], set[str], str]:
    root = ET.parse(urdf_path).getroot()
    actuated_joints: list[str] = []
    joint_limits: dict[str, dict[str, float]] = {}
    joint_child_links: dict[str, str] = {}
    link_names = {link.attrib["name"] for link in root.findall("link")}
    child_links = set()

    for joint in root.findall("joint"):
        name = joint.attrib["name"]
        joint_type = joint.attrib.get("type", "fixed")
        child = joint.find("child")
        if child is not None:
            child_link = child.attrib["link"]
            joint_child_links[name] = child_link
            child_links.add(child_link)
        if joint_type in {"fixed", "floating", "planar"}:
            continue

        actuated_joints.append(name)
        limit = joint.find("limit")
        if limit is not None:
            joint_limits[name] = {
                "min_position": float(limit.attrib.get("lower", "-3.14")),
                "max_position": float(limit.attrib.get("upper", "3.14")),
                "max_velocity": float(limit.attrib.get("velocity", "1.0")),
                "max_effort": float(limit.attrib.get("effort", "1.0")),
            }
        else:
            joint_limits[name] = {
                "min_position": -3.14,
                "max_position": 3.14,
                "max_velocity": 1.0,
                "max_effort": 1.0,
            }

    root_links = sorted(link_names - child_links)
    if len(root_links) != 1:
        raise ValueError(f"Expected exactly one URDF root link, found: {root_links}")

    return actuated_joints, joint_limits, joint_child_links, link_names, root_links[0]


def _pick_default_arm_joints(actuated_joints: list[str]) -> list[str]:
    preferred = [
        "right_base_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]
    if all(joint in actuated_joints for joint in preferred):
        return preferred
    return [joint for joint in actuated_joints if joint.startswith("right_") and "gripper" not in joint][:6]


def _pick_default_gripper_joints(actuated_joints: list[str]) -> list[str]:
    preferred = ["right_gripper1_joint", "right_gripper2_joint"]
    if all(joint in actuated_joints for joint in preferred):
        return preferred
    return [joint for joint in actuated_joints if joint.startswith("right_gripper")]


def _resolve_joint_names(
    actuated_joints: list[str],
    requested: str | None,
    defaults: list[str],
    label: str,
) -> list[str]:
    if requested:
        resolved = [joint.strip() for joint in requested.split(",") if joint.strip()]
    else:
        resolved = defaults
    missing = [joint for joint in resolved if joint not in actuated_joints]
    if missing:
        raise ValueError(f"Requested {label} joints are not in the URDF: {missing}")
    if not resolved:
        raise ValueError(f"No {label} joints resolved from the URDF.")
    return resolved


def _shrink_placeholder_arm_limits(
    joint_limits: dict[str, dict[str, float]],
    arm_joints: list[str],
    shrink_ratio: float = 0.95,
) -> dict[str, dict[str, float]]:
    adjusted_limits = {joint_name: dict(limit) for joint_name, limit in joint_limits.items()}
    generic_lower = -3.14
    generic_upper = 3.14
    tolerance = 1e-3

    for joint_name in arm_joints:
        limit = adjusted_limits.get(joint_name)
        if limit is None:
            continue

        lower = float(limit.get("min_position", generic_lower))
        upper = float(limit.get("max_position", generic_upper))
        is_generic = abs(lower - generic_lower) < tolerance and abs(upper - generic_upper) < tolerance
        if not is_generic:
            continue

        midpoint = 0.5 * (lower + upper)
        half_range = 0.5 * (upper - lower) * shrink_ratio
        limit["min_position"] = midpoint - half_range
        limit["max_position"] = midpoint + half_range
        limit["limit_source"] = "generic_urdf_shrunk_5pct"

    return adjusted_limits


def _apply_reference_right_arm_limits(
    joint_limits: dict[str, dict[str, float]],
    arm_joints: list[str],
) -> dict[str, dict[str, float]]:
    adjusted_limits = {joint_name: dict(limit) for joint_name, limit in joint_limits.items()}

    # Fallback reference from Fourier GR-2 official arm joint-limit table.
    # Mapping inference for this custom URDF:
    # right_base_pitch_joint -> right_shoulder_pitch_joint
    # right_shoulder_roll_joint -> right_shoulder_roll_joint
    # right_shoulder_yaw_joint -> right_shoulder_yaw_joint
    # right_elbow_pitch_joint -> right_elbow_pitch_joint
    # right_wrist_yaw_joint -> right_wrist_yaw_joint
    # right_wrist_pitch_joint -> right_wrist_pitch_joint
    # Source:
    # https://support.fftai.com/en/docs/GR-X-Humanoid-Robot/GR2/GR-2_Introduction/
    gr2_reference_limits = {
        "right_base_pitch_joint": (-2.9671, 2.9671),
        "right_shoulder_roll_joint": (-2.7925, 0.5236),
        "right_shoulder_yaw_joint": (-1.8326, 1.8326),
        "right_elbow_pitch_joint": (-1.5272, 0.47997),
        "right_wrist_yaw_joint": (-1.8326, 1.8326),
        "right_wrist_pitch_joint": (-0.61087, 0.61087),
    }

    tolerance = 1e-3
    generic_lower = -3.14
    generic_upper = 3.14

    for joint_name in arm_joints:
        if joint_name not in gr2_reference_limits:
            continue
        limit = adjusted_limits.get(joint_name)
        if limit is None:
            continue

        lower = float(limit.get("min_position", generic_lower))
        upper = float(limit.get("max_position", generic_upper))
        is_generic = abs(lower - generic_lower) < tolerance and abs(upper - generic_upper) < tolerance
        if not is_generic:
            continue

        ref_lower, ref_upper = gr2_reference_limits[joint_name]
        limit["min_position"] = ref_lower
        limit["max_position"] = ref_upper
        limit["limit_source"] = "fourier_gr2_reference_mapped"
        limit["limit_reference"] = "https://support.fftai.com/en/docs/GR-X-Humanoid-Robot/GR2/GR-2_Introduction/"

    return adjusted_limits


def load_robot_training_config(
    repo_root: Path | None = None,
    robot_description_path: Path | None = None,
    arm_joints: str | None = None,
    gripper_joints: str | None = None,
    end_effector_link: str | None = None,
) -> RobotTrainingConfig:
    resolved_repo_root = _resolve_repo_root(repo_root)
    resolved_robot_description_path = _resolve_robot_description_path(
        resolved_repo_root, robot_description_path
    )
    urdf_path = _resolve_urdf_path(resolved_robot_description_path)
    runtime_urdf_path = _rewrite_package_meshes(urdf_path, resolved_robot_description_path)

    actuated_joints, joint_limits, joint_child_links, link_names, urdf_root_link = _parse_urdf_joints(urdf_path)
    resolved_arm_joints = _resolve_joint_names(
        actuated_joints, arm_joints, _pick_default_arm_joints(actuated_joints), "arm"
    )
    resolved_gripper_joints = _resolve_joint_names(
        actuated_joints, gripper_joints, _pick_default_gripper_joints(actuated_joints), "gripper"
    )
    joint_limits = _apply_reference_right_arm_limits(joint_limits, resolved_arm_joints)
    joint_limits = _shrink_placeholder_arm_limits(joint_limits, resolved_arm_joints)

    if end_effector_link is None:
        end_effector_link = joint_child_links.get(resolved_arm_joints[-1], "right_wrist_yaw_link")
    if end_effector_link not in link_names:
        raise ValueError(f"End-effector link '{end_effector_link}' not found in the URDF.")

    return RobotTrainingConfig(
        repo_root=resolved_repo_root,
        robot_description_path=resolved_robot_description_path,
        urdf_path=urdf_path,
        runtime_urdf_path=runtime_urdf_path,
        urdf_root_link=urdf_root_link,
        arm_joints=resolved_arm_joints,
        gripper_joints=resolved_gripper_joints,
        joint_limits=joint_limits,
        end_effector_link=end_effector_link,
        # URDF frame convention for this robot: +Y is forward, +X is left/right.
        # Keep the brick centered laterally and sample it forward on the table.
        train_brick_range={"x_min": -0.16, "x_max": 0.16, "y_min": 0.50, "y_max": 0.64},
        eval_brick_range={"x_min": -0.20, "x_max": 0.20, "y_min": 0.46, "y_max": 0.68},
    )
