from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Empty, Float32MultiArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


@dataclass
class RobotObservation:
    arm_positions: np.ndarray
    arm_velocities: np.ndarray
    gripper_positions: np.ndarray
    brick_position: np.ndarray
    ee_position: np.ndarray
    grasp_success: bool


class IsaacRosBridge(Node):
    def __init__(self, arm_joints: list[str], gripper_joints: list[str]) -> None:
        super().__init__("isaac_rl_bridge_client")
        self._arm_joints = arm_joints
        self._gripper_joints = gripper_joints
        self._joint_state_lock = threading.Lock()
        self._latest_joint_state: Optional[JointState] = None
        self._latest_brick_pose: Optional[PoseStamped] = None
        self._latest_ee_pose: Optional[PoseStamped] = None
        self._grasp_success = False

        self._joint_command_pub = self.create_publisher(
            JointTrajectory, "/isaac_rl/joint_command", 10
        )
        self._gripper_command_pub = self.create_publisher(
            Float32MultiArray, "/isaac_rl/gripper_command", 10
        )
        self._reset_pub = self.create_publisher(Empty, "/isaac_rl/reset", 10)

        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 20)
        self.create_subscription(PoseStamped, "/isaac_rl/brick_pose", self._brick_pose_cb, 20)
        self.create_subscription(PoseStamped, "/isaac_rl/end_effector_pose", self._ee_pose_cb, 20)
        self.create_subscription(Bool, "/isaac_rl/grasp_success", self._grasp_success_cb, 20)

    def _joint_state_cb(self, msg: JointState) -> None:
        with self._joint_state_lock:
            self._latest_joint_state = msg

    def _brick_pose_cb(self, msg: PoseStamped) -> None:
        self._latest_brick_pose = msg

    def _ee_pose_cb(self, msg: PoseStamped) -> None:
        self._latest_ee_pose = msg

    def _grasp_success_cb(self, msg: Bool) -> None:
        self._grasp_success = msg.data

    def reset_episode(self) -> None:
        self._grasp_success = False
        self._reset_pub.publish(Empty())

    def send_arm_command(self, target_positions: np.ndarray, duration_sec: float = 0.2) -> None:
        msg = JointTrajectory()
        msg.joint_names = list(self._arm_joints)
        point = JointTrajectoryPoint()
        point.positions = [float(x) for x in target_positions]
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        msg.points = [point]
        self._joint_command_pub.publish(msg)

    def send_gripper_command(self, target_positions: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = [float(x) for x in target_positions]
        self._gripper_command_pub.publish(msg)

    def get_observation(self) -> Optional[RobotObservation]:
        if self._latest_brick_pose is None or self._latest_ee_pose is None:
            return None
        with self._joint_state_lock:
            joint_state = self._latest_joint_state
        if joint_state is None:
            return None

        index = {name: i for i, name in enumerate(joint_state.name)}
        arm_positions = np.array([joint_state.position[index[j]] for j in self._arm_joints], dtype=np.float32)
        arm_velocities = np.array(
            [joint_state.velocity[index[j]] if joint_state.velocity else 0.0 for j in self._arm_joints],
            dtype=np.float32,
        )
        gripper_positions = np.array(
            [joint_state.position[index[j]] for j in self._gripper_joints],
            dtype=np.float32,
        )
        brick_position = np.array(
            [
                self._latest_brick_pose.pose.position.x,
                self._latest_brick_pose.pose.position.y,
                self._latest_brick_pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        ee_position = np.array(
            [
                self._latest_ee_pose.pose.position.x,
                self._latest_ee_pose.pose.position.y,
                self._latest_ee_pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        return RobotObservation(
            arm_positions=arm_positions,
            arm_velocities=arm_velocities,
            gripper_positions=gripper_positions,
            brick_position=brick_position,
            ee_position=ee_position,
            grasp_success=self._grasp_success,
        )


class Ros2SpinThread:
    def __init__(self, node: Node) -> None:
        self._node = node
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def shutdown(self) -> None:
        self._executor.shutdown()
        self._node.destroy_node()


def init_ros() -> None:
    if not rclpy.ok():
        rclpy.init()
