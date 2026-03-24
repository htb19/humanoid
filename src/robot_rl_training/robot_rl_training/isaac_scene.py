from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np

if __package__ in (None, ""):
    this_file = Path(__file__).resolve()
    package_root = this_file.parent.parent
    workspace_root = package_root.parent.parent
    sys.path.insert(0, str(package_root.parent))
    sys.path.insert(0, str(workspace_root / "src"))
    from robot_rl_training.config import load_robot_training_config
else:
    from .config import load_robot_training_config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isaac Sim brick-pick scene for PPO training.")
    parser.add_argument("--headless", default="false")
    parser.add_argument("--brick-x-min", type=float, default=0.30)
    parser.add_argument("--brick-x-max", type=float, default=0.45)
    parser.add_argument("--brick-y-min", type=float, default=-0.18)
    parser.add_argument("--brick-y-max", type=float, default=0.18)
    parser.add_argument("--eval-brick-x-min", type=float, default=0.35)
    parser.add_argument("--eval-brick-x-max", type=float, default=0.50)
    parser.add_argument("--eval-brick-y-min", type=float, default=-0.22)
    parser.add_argument("--eval-brick-y-max", type=float, default=0.22)
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--workspace-root", type=Path, default=None)
    parser.add_argument("--ros-package-path", type=Path, default=None)
    return parser


def main() -> None:
    args, unknown = _build_arg_parser().parse_known_args()
    headless = str(args.headless).lower() == "true"

    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp

    simulation_app = SimulationApp({"headless": headless})

    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Empty, Float32MultiArray
        from trajectory_msgs.msg import JointTrajectory

        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.controllers import ArticulationController
        from omni.isaac.core.objects import DynamicCuboid
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.prims import get_prim_at_path
        from omni.isaac.core.utils.types import ArticulationAction
        from pxr import UsdGeom

    except Exception:
        simulation_app.close()
        raise

    enable_extension("omni.isaac.ros2_bridge")
    enable_extension("omni.importer.urdf")

    if not rclpy.ok():
        rclpy.init(args=unknown)

    cfg = load_robot_training_config()
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()

    from omni import kit, usd
    import omni
    import omni.kit.commands

    result, imported_prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=str(cfg.urdf_path),
        import_config={
            "merge_fixed_joints": False,
            "fix_base": True,
            "make_default_prim": False,
            "self_collision": False,
            "create_physics_scene": True,
        },
    )
    if not result:
        raise RuntimeError(f"Failed to import URDF: {cfg.urdf_path}")
    robot_prim_path = imported_prim_path

    brick_prim_path = "/World/toy_brick"
    brick = world.scene.add(
        DynamicCuboid(
            prim_path=brick_prim_path,
            name="toy_brick",
            position=np.array([0.38, 0.0, 0.025], dtype=np.float32),
            scale=np.array([0.04, 0.02, 0.02], dtype=np.float32),
            color=np.array([0.8, 0.1, 0.1], dtype=np.float32),
            mass=0.05,
        )
    )

    articulation = Articulation(prim_path=robot_prim_path, name="humanoid")
    controller = ArticulationController()

    ros_node = rclpy.create_node("isaac_rl_scene")
    brick_pose_pub = ros_node.create_publisher(PoseStamped, "/isaac_rl/brick_pose", 10)
    ee_pose_pub = ros_node.create_publisher(PoseStamped, "/isaac_rl/end_effector_pose", 10)
    joint_state_pub = ros_node.create_publisher(JointState, "/joint_states", 10)
    grasp_success_pub = ros_node.create_publisher(Bool, "/isaac_rl/grasp_success", 10)

    arm_joint_targets = np.zeros(len(cfg.arm_joints), dtype=np.float32)
    gripper_targets = np.zeros(len(cfg.gripper_joints), dtype=np.float32)
    pending_reset = False

    def _sample_brick_xy() -> tuple[float, float]:
        if args.evaluation:
            return (
                random.uniform(args.eval_brick_x_min, args.eval_brick_x_max),
                random.uniform(args.eval_brick_y_min, args.eval_brick_y_max),
            )
        return (
            random.uniform(args.brick_x_min, args.brick_x_max),
            random.uniform(args.brick_y_min, args.brick_y_max),
        )

    def _joint_command_cb(msg: JointTrajectory) -> None:
        nonlocal arm_joint_targets
        if msg.points:
            arm_joint_targets = np.array(msg.points[0].positions, dtype=np.float32)

    def _gripper_command_cb(msg: Float32MultiArray) -> None:
        nonlocal gripper_targets
        gripper_targets = np.array(msg.data, dtype=np.float32)

    def _reset_cb(_: Empty) -> None:
        nonlocal pending_reset
        pending_reset = True

    ros_node.create_subscription(JointTrajectory, "/isaac_rl/joint_command", _joint_command_cb, 10)
    ros_node.create_subscription(Float32MultiArray, "/isaac_rl/gripper_command", _gripper_command_cb, 10)
    ros_node.create_subscription(Empty, "/isaac_rl/reset", _reset_cb, 10)

    world.reset()
    articulation.initialize()
    controller.initialize(articulation)

    dof_names = articulation.dof_names
    arm_indices = [dof_names.index(name) for name in cfg.arm_joints]
    gripper_indices = [dof_names.index(name) for name in cfg.gripper_joints]
    ee_prim = get_prim_at_path(f"{robot_prim_path}/{cfg.end_effector_parent_link}")

    def _reset_scene() -> None:
        x, y = _sample_brick_xy()
        brick.set_world_pose(position=np.array([x, y, 0.025], dtype=np.float32))
        brick.set_linear_velocity(np.zeros(3, dtype=np.float32))
        brick.set_angular_velocity(np.zeros(3, dtype=np.float32))
        articulation.set_joint_positions(np.zeros(len(dof_names), dtype=np.float32))
        articulation.set_joint_velocities(np.zeros(len(dof_names), dtype=np.float32))

    def _publish_state() -> None:
        joint_positions = articulation.get_joint_positions()
        joint_velocities = articulation.get_joint_velocities()

        joint_state = JointState()
        joint_state.header.stamp = ros_node.get_clock().now().to_msg()
        joint_state.name = list(dof_names)
        joint_state.position = [float(x) for x in joint_positions]
        joint_state.velocity = [float(x) for x in joint_velocities]
        joint_state_pub.publish(joint_state)

        brick_pose = brick.get_world_pose()[0]
        brick_msg = PoseStamped()
        brick_msg.header.stamp = ros_node.get_clock().now().to_msg()
        brick_msg.header.frame_id = "world"
        brick_msg.pose.position.x = float(brick_pose[0])
        brick_msg.pose.position.y = float(brick_pose[1])
        brick_msg.pose.position.z = float(brick_pose[2])
        brick_pose_pub.publish(brick_msg)

        ee_tf = UsdGeom.Xformable(ee_prim).ComputeLocalToWorldTransform(0.0)
        ee_translation = ee_tf.ExtractTranslation()
        ee_msg = PoseStamped()
        ee_msg.header.stamp = ros_node.get_clock().now().to_msg()
        ee_msg.header.frame_id = "world"
        ee_msg.pose.position.x = float(ee_translation[0])
        ee_msg.pose.position.y = float(ee_translation[1])
        ee_msg.pose.position.z = float(ee_translation[2])
        ee_pose_pub.publish(ee_msg)

        grasp_success = Bool()
        horizontal_distance = np.linalg.norm(
            np.array([brick_pose[0] - ee_translation[0], brick_pose[1] - ee_translation[1]], dtype=np.float32)
        )
        gripper_closed = float(np.mean(gripper_targets)) > 0.02
        grasp_success.data = bool(horizontal_distance < 0.04 and gripper_closed and brick_pose[2] > 0.03)
        grasp_success_pub.publish(grasp_success)

    _reset_scene()
    while simulation_app.is_running():
        rclpy.spin_once(ros_node, timeout_sec=0.0)
        if pending_reset:
            _reset_scene()
            pending_reset = False

        current_positions = articulation.get_joint_positions()
        target_positions = current_positions.copy()
        target_positions[arm_indices] = arm_joint_targets
        target_positions[gripper_indices] = gripper_targets
        controller.apply_action(ArticulationAction(joint_positions=target_positions))

        world.step(render=not headless)
        _publish_state()

    ros_node.destroy_node()
    rclpy.shutdown()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"isaac_scene.py failed: {exc}", file=sys.stderr)
        raise
