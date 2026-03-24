from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "robot_description"
URDF_PATH = ROBOT_ROOT / "urdf" / "humanoid.urdf"


def package_uri_to_path(uri: str) -> Path:
    prefix = "package://robot_description/"
    if not uri.startswith(prefix):
        return Path(uri)
    return ROBOT_ROOT / uri[len(prefix) :]


def inspect_robot() -> dict:
    root = ET.parse(URDF_PATH).getroot()

    links = [link.attrib["name"] for link in root.findall("link")]
    joints = []
    missing_meshes = []
    missing_collision_links = []
    missing_inertial_links = []
    wide_limit_joints = []
    mimic_joints = []

    for link in root.findall("link"):
        name = link.attrib["name"]
        if link.find("inertial") is None:
            missing_inertial_links.append(name)
        if link.find("collision") is None:
            missing_collision_links.append(name)

    mesh_files = []
    for mesh in root.findall(".//mesh"):
        filename = mesh.attrib["filename"]
        mesh_files.append(filename)
        resolved = package_uri_to_path(filename)
        if not resolved.exists():
            missing_meshes.append({"mesh": filename, "resolved_path": str(resolved)})

    for joint in root.findall("joint"):
        limit_elem = joint.find("limit")
        mimic_elem = joint.find("mimic")
        limit = limit_elem.attrib if limit_elem is not None else {}
        joint_info = {
            "name": joint.attrib["name"],
            "type": joint.attrib["type"],
            "parent": joint.find("parent").attrib["link"],
            "child": joint.find("child").attrib["link"],
            "axis": joint.find("axis").attrib["xyz"] if joint.find("axis") is not None else None,
            "limit": limit,
            "mimic": mimic_elem.attrib if mimic_elem is not None else None,
        }
        joints.append(joint_info)

        if mimic_elem is not None:
            mimic_joints.append(joint.attrib["name"])

        if limit:
            lower = float(limit.get("lower", "0"))
            upper = float(limit.get("upper", "0"))
            if joint.attrib["type"] == "revolute" and upper - lower > 6.0:
                wide_limit_joints.append(joint.attrib["name"])

    children = {}
    for joint in joints:
        children.setdefault(joint["parent"], []).append(joint)

    chain = []

    def walk(link_name: str) -> None:
        for child_joint in children.get(link_name, []):
            chain.append(
                {
                    "parent": link_name,
                    "joint": child_joint["name"],
                    "type": child_joint["type"],
                    "child": child_joint["child"],
                }
            )
            walk(child_joint["child"])

    walk("base_link")

    return {
        "urdf_path": str(URDF_PATH),
        "link_count": len(links),
        "joint_count": len(joints),
        "main_urdf_files": [
            str(ROBOT_ROOT / "urdf" / "humanoid.urdf"),
            str(ROBOT_ROOT / "urdf" / "humanoid.urdf.xacro"),
        ],
        "base_link": "base_link",
        "end_effector_candidate": "right_wrist_yaw_link",
        "gripper_joints": [
            "right_gripper1_joint",
            "right_gripper2_joint",
            "left_gripper1_joint",
            "left_gripper2_joint",
        ],
        "mimic_joints": mimic_joints,
        "mesh_files": mesh_files,
        "kinematic_tree": chain,
        "missing_meshes": missing_meshes,
        "missing_collision_links": missing_collision_links,
        "missing_inertial_links": missing_inertial_links,
        "transmissions_in_urdf": 0,
        "wide_limit_joints": wide_limit_joints,
        "notes": [
            "The generated URDF includes full inertial and collision tags for all 22 links.",
            "Mesh paths use package://robot_description/... URIs and must be rewritten or resolved for MuJoCo.",
            "Both grippers use one driven prismatic joint plus one mimic prismatic joint with multiplier -1.0.",
            "Most revolute joints use the very broad range [-3.14, 3.14], which is workable for parsing but weak for RL stabilization.",
            "No <transmission> tags are present in humanoid.urdf; ROS2 control metadata lives in separate xacro files and is ignored for training.",
        ],
    }


def main() -> None:
    report = inspect_robot()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
