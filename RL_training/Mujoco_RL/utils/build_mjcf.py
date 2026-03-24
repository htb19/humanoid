from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROBOT_ROOT = PROJECT_ROOT / "robot_description"
URDF_PATH = ROBOT_ROOT / "urdf" / "humanoid.urdf"
ASSETS_DIR = PROJECT_ROOT / "assets"
ROBOT_XML_PATH = ASSETS_DIR / "humanoid_right_arm.xml"
SCENE_XML_PATH = ASSETS_DIR / "pick_brick_scene.xml"

ACTIVE_CHAIN = [
    "right_base_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_gripper1_joint",
    "right_gripper2_joint",
]

LINK_ORDER = [
    "base_link",
    "right_base_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_pitch_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_gripper1_link",
    "right_gripper2_link",
]

JOINT_RANGE_OVERRIDES = {
    "right_base_pitch_joint": (-1.6, 1.6),
    "right_shoulder_roll_joint": (-1.4, 1.4),
    "right_shoulder_yaw_joint": (-1.8, 1.8),
    "right_elbow_pitch_joint": (-2.0, 2.0),
    "right_wrist_pitch_joint": (-1.8, 1.8),
    "right_wrist_yaw_joint": (-2.5, 2.5),
    "right_gripper1_joint": (0.0, 0.028),
    "right_gripper2_joint": (-0.028, 0.0),
}

ARM_KP = {
    "right_base_pitch_joint": 80,
    "right_shoulder_roll_joint": 80,
    "right_shoulder_yaw_joint": 70,
    "right_elbow_pitch_joint": 70,
    "right_wrist_pitch_joint": 40,
    "right_wrist_yaw_joint": 35,
    "right_gripper1_joint": 250,
    "right_gripper2_joint": 250,
}

LINK_COLLISION_GEOMS = {
    "base_link": '<geom name="base_collision" type="box" pos="0 0.03 0.09" size="0.16 0.11 0.16" rgba="0.7 0.75 0.82 1"/>',
    "right_base_pitch_link": '<geom name="right_base_pitch_collision" type="capsule" fromto="0 0 0 -0.12 0 0.08" size="0.055" rgba="0.8 0.8 0.8 1"/>',
    "right_shoulder_roll_link": '<geom name="right_shoulder_roll_collision" type="capsule" fromto="0 0 0 -0.15 0 -0.06" size="0.045" rgba="0.8 0.8 0.8 1"/>',
    "right_shoulder_yaw_link": '<geom name="right_shoulder_yaw_collision" type="capsule" fromto="0 0 0 0.0 0.07 -0.10" size="0.04" rgba="0.8 0.8 0.8 1"/>',
    "right_elbow_pitch_link": '<geom name="right_elbow_pitch_collision" type="capsule" fromto="0 0 0 0.0 0.12 -0.05" size="0.032" rgba="0.8 0.8 0.8 1"/>',
    "right_wrist_pitch_link": '<geom name="right_wrist_pitch_collision" type="capsule" fromto="0 0 0 0.0 0.04 0.11" size="0.028" rgba="0.8 0.8 0.8 1"/>',
    "right_wrist_yaw_link": '<geom name="right_wrist_yaw_collision" type="capsule" fromto="0 0 0 0.0 0.12 -0.03" size="0.026" rgba="0.8 0.8 0.8 1"/>',
    "right_gripper1_link": '<geom name="finger_left_pad" type="box" pos="-0.055 0 0.024" size="0.035 0.008 0.012" friction="1.5 0.02 0.001" rgba="0.15 0.15 0.15 1"/>',
    "right_gripper2_link": '<geom name="finger_right_pad" type="box" pos="0.055 0 0.024" size="0.035 0.008 0.012" friction="1.5 0.02 0.001" rgba="0.15 0.15 0.15 1"/>',
}


def _load_urdf() -> tuple[dict, dict]:
    root = ET.parse(URDF_PATH).getroot()
    links = {link.attrib["name"]: link for link in root.findall("link")}
    joints = {joint.attrib["name"]: joint for joint in root.findall("joint")}
    return links, joints


def _mesh_filename(link_name: str, links: dict) -> str | None:
    link = links[link_name]
    mesh = link.find("visual/geometry/mesh")
    if mesh is None:
        return None
    filename = mesh.attrib["filename"].replace("package://robot_description/meshes/", "")
    return filename


def _inertial_xml(link_name: str, links: dict, indent: str) -> str:
    inertial = links[link_name].find("inertial")
    if inertial is None:
        return ""
    origin = inertial.find("origin")
    mass = inertial.find("mass")
    inertia = inertial.find("inertia")
    return (
        f'{indent}<inertial pos="{origin.attrib["xyz"]}" mass="{mass.attrib["value"]}" '
        f'fullinertia="{inertia.attrib["ixx"]} {inertia.attrib["iyy"]} {inertia.attrib["izz"]} '
        f'{inertia.attrib["ixy"]} {inertia.attrib["ixz"]} {inertia.attrib["iyz"]}"/>\n'
    )


def _body_visual_xml(link_name: str, links: dict, indent: str) -> str:
    filename = _mesh_filename(link_name, links)
    if filename is None:
        return ""
    return (
        f'{indent}<geom name="{link_name}_visual" type="mesh" mesh="{link_name}_mesh" '
        'contype="0" conaffinity="0" group="1" rgba="0.9 0.9 0.9 1"/>\n'
    )


def _joint_xml(joint_name: str, joints: dict, indent: str) -> str:
    joint = joints[joint_name]
    low, high = JOINT_RANGE_OVERRIDES[joint_name]
    joint_type = "hinge" if joint.attrib["type"] == "revolute" else "slide"
    damping = "3.0" if joint_type == "hinge" else "1.0"
    armature = "0.02" if joint_type == "hinge" else "0.001"
    return (
        f'{indent}<joint name="{joint_name}" type="{joint_type}" axis="{joint.find("axis").attrib["xyz"]}" '
        f'range="{low} {high}" damping="{damping}" armature="{armature}" limited="true"/>\n'
    )


def _build_body(link_name: str, joints: dict, links: dict, child_map: dict, indent: str) -> str:
    out = []
    out.append(_inertial_xml(link_name, links, indent))
    out.append(_body_visual_xml(link_name, links, indent))
    collision = LINK_COLLISION_GEOMS.get(link_name)
    if collision:
        out.append(f"{indent}{collision}\n")
    if link_name == "right_wrist_yaw_link":
        out.append(
            f'{indent}<site name="ee_site" type="sphere" pos="0 0.145 0.024" size="0.012" rgba="1 0 0 1"/>\n'
        )
        out.append(
            f'{indent}<site name="grasp_target_site" type="sphere" pos="0 0.165 0.024" size="0.008" rgba="0 1 0 1"/>\n'
        )

    for child_joint_name in child_map.get(link_name, []):
        child_joint = joints[child_joint_name]
        child_link = child_joint.find("child").attrib["link"]
        origin = child_joint.find("origin")
        pos = origin.attrib["xyz"]
        euler = origin.attrib["rpy"]
        out.append(f'{indent}<body name="{child_link}" pos="{pos}" euler="{euler}">\n')
        out.append(_joint_xml(child_joint_name, joints, indent + "  "))
        out.append(_build_body(child_link, joints, links, child_map, indent + "  "))
        out.append(f"{indent}</body>\n")
    return "".join(out)


def build_robot_xml() -> str:
    links, joints = _load_urdf()
    child_map = {}
    for joint_name in ACTIVE_CHAIN:
        parent = joints[joint_name].find("parent").attrib["link"]
        child_map.setdefault(parent, []).append(joint_name)

    mesh_assets = []
    for link_name in LINK_ORDER:
        filename = _mesh_filename(link_name, links)
        if filename is None:
            continue
        mesh_assets.append(f'    <mesh name="{link_name}_mesh" file="{filename}"/>')

    actuators = []
    for joint_name in ACTIVE_CHAIN:
        low, high = JOINT_RANGE_OVERRIDES[joint_name]
        actuators.append(
            f'    <position name="{joint_name}_act" joint="{joint_name}" kp="{ARM_KP[joint_name]}" ctrlrange="{low} {high}" forcelimited="false"/>'
        )

    body_tree = _build_body("base_link", joints, links, child_map, "      ")

    return f"""<mujoco model="humanoid_right_arm">
  <compiler angle="radian" coordinate="local" meshdir="../robot_description/meshes" autolimits="true"/>
  <option timestep="0.004" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>
  <visual>
    <headlight ambient="0.55 0.55 0.55" diffuse="0.4 0.4 0.4" specular="0.1 0.1 0.1"/>
    <rgba haze="0.15 0.2 0.25 1"/>
  </visual>
  <default>
    <default class="visual">
      <geom group="1" type="mesh" contype="0" conaffinity="0"/>
    </default>
    <default class="collision">
      <geom group="3" density="900"/>
    </default>
  </default>
  <asset>
{chr(10).join(mesh_assets)}
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 2.4" dir="0 0 -1" directional="true"/>
    <camera name="overview" pos="1.25 -1.1 1.35" xyaxes="0.72 0.69 0 -0.35 0.36 0.86"/>
    <geom name="ground" type="plane" size="2 2 0.1" rgba="0.92 0.94 0.96 1" friction="1 0.1 0.01"/>
    <geom name="table" type="box" pos="0.28 0.30 0.725" size="0.25 0.20 0.025" rgba="0.60 0.44 0.32 1" friction="1.1 0.05 0.01"/>
    <body name="brick" pos="0.28 0.27 0.77">
      <freejoint name="brick_freejoint"/>
      <geom name="brick_geom" type="box" size="0.02 0.01 0.01" mass="0.08" friction="1.2 0.02 0.002" rgba="0.85 0.25 0.18 1"/>
      <site name="brick_site" type="sphere" size="0.008" rgba="1 1 0 1"/>
    </body>
    <body name="base_link" pos="0 0 0.6">
{body_tree}    </body>
  </worldbody>
  <actuator>
{chr(10).join(actuators)}
  </actuator>
  <keyframe>
    <key name="home" qpos="0 0 0 0 0 0 0.025 -0.025 0.28 0.27 0.77 1 0 0 0"/>
  </keyframe>
</mujoco>
"""


def build_scene_xml() -> str:
    return """<mujoco model="pick_brick_scene">
  <include file="humanoid_right_arm.xml"/>
</mujoco>
"""


def ensure_assets_built(force: bool = False) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    if force or not ROBOT_XML_PATH.exists():
        ROBOT_XML_PATH.write_text(build_robot_xml())
    if force or not SCENE_XML_PATH.exists():
        SCENE_XML_PATH.write_text(build_scene_xml())


def main() -> None:
    ensure_assets_built(force=True)
    print(f"Wrote {ROBOT_XML_PATH}")
    print(f"Wrote {SCENE_XML_PATH}")


if __name__ == "__main__":
    main()
