#!/usr/bin/env python3
from __future__ import annotations

import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from typing import Dict, List, Optional

import rclpy
from control_msgs.msg import JointJog
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_srvs.srv import Trigger


@dataclass(frozen=True)
class JointCommand:
    target: str
    joint_name: str
    velocity: float


class KeyboardServoControl(Node):
    HEAD_JOINTS = ["neck_pitch_joint", "neck_yaw_joint"]
    LEFT_ARM_JOINTS = [
        "left_base_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
    ]
    RIGHT_ARM_JOINTS = [
        "right_base_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ]

    def __init__(self) -> None:
        super().__init__("keyboard_servo_control")

        self.declare_parameter(
            "head_joint_topic",
            "/head_servo/moveit_servo/delta_joint_cmds",
        )
        self.declare_parameter(
            "left_joint_topic",
            "/left_servo/moveit_servo/delta_joint_cmds",
        )
        self.declare_parameter(
            "right_joint_topic",
            "/right_servo/moveit_servo/delta_joint_cmds",
        )
        self.declare_parameter(
            "head_start_service",
            "/head_servo/moveit_servo/start_servo",
        )
        self.declare_parameter(
            "left_start_service",
            "/left_servo/moveit_servo/start_servo",
        )
        self.declare_parameter(
            "right_start_service",
            "/right_servo/moveit_servo/start_servo",
        )
        self.declare_parameter("command_frame", "base_link")
        self.declare_parameter("head_joint_speed", 1.0)
        self.declare_parameter("arm_joint_speed", 0.8)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("command_timeout", 0.2)
        self.declare_parameter("wait_service_timeout_sec", 30.0)

        self.command_frame = self.get_string_param("command_frame")
        self.head_joint_speed = self.get_double_param("head_joint_speed")
        self.arm_joint_speed = self.get_double_param("arm_joint_speed")
        self.publish_rate = self.get_double_param("publish_rate")
        self.command_timeout = self.get_double_param("command_timeout")
        self.wait_service_timeout_sec = self.get_double_param(
            "wait_service_timeout_sec"
        )

        self.joints_by_target: Dict[str, List[str]] = {
            "head": self.HEAD_JOINTS,
            "left": self.LEFT_ARM_JOINTS,
            "right": self.RIGHT_ARM_JOINTS,
        }
        self._publishers: Dict[str, Publisher] = {
            "head": self.create_publisher(
                JointJog,
                self.get_string_param("head_joint_topic"),
                10,
            ),
            "left": self.create_publisher(
                JointJog,
                self.get_string_param("left_joint_topic"),
                10,
            ),
            "right": self.create_publisher(
                JointJog,
                self.get_string_param("right_joint_topic"),
                10,
            ),
        }
        self.start_services = {
            "head": self.get_string_param("head_start_service"),
            "left": self.get_string_param("left_start_service"),
            "right": self.get_string_param("right_start_service"),
        }

        self.active_arm = "left"
        self.direction = 1.0
        self.current_command: Optional[JointCommand] = None
        self.last_key_time = 0.0
        self.running = True
        self.zero_sent = True
        self.lock = threading.Lock()

        self.start_all_servos()

        self.keyboard_thread = threading.Thread(
            target=self.keyboard_loop,
            daemon=True,
        )
        self.keyboard_thread.start()

        self.timer = self.create_timer(
            1.0 / max(self.publish_rate, 1.0),
            self.publish_current_command,
        )

        self.print_help()

    def get_string_param(self, name: str) -> str:
        return self.get_parameter(name).get_parameter_value().string_value

    def get_double_param(self, name: str) -> float:
        return self.get_parameter(name).get_parameter_value().double_value

    def start_all_servos(self) -> None:
        for target, service_name in self.start_services.items():
            self.start_servo(target, service_name)

    def start_servo(self, target: str, service_name: str) -> None:
        client = self.create_client(Trigger, service_name)

        self.get_logger().info(f"[{target}] waiting for servo service: {service_name}")
        if not client.wait_for_service(timeout_sec=self.wait_service_timeout_sec):
            raise RuntimeError(f"[{target}] servo start service unavailable")

        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=self.wait_service_timeout_sec,
        )

        if not future.done() or future.result() is None:
            raise RuntimeError(f"[{target}] failed to call servo start service")

        result = future.result()
        if result is None:
            raise RuntimeError("Servo service returned no response")
        if not result.success:
            raise RuntimeError(f"[{target}] servo failed to start: {result.message}")

        self.get_logger().info(f"[{target}] servo started: {result.message}")

    def print_help(self) -> None:
        help_text = (
            "\n"
            "Keyboard servo control ready.\n"
            "Head joint control:\n"
            "  w/s: neck_pitch_joint positive/negative\n"
            "  a/d: neck_yaw_joint positive/negative\n"
            "Arm joint control:\n"
            "  1-6: jog selected arm joint\n"
            "  Tab: switch selected arm between left/right\n"
            "  r: reverse arm jog direction\n"
            "Other:\n"
            "  space: stop all\n"
            "  q: quit node\n"
        )
        self.get_logger().info(help_text)

    def keyboard_loop(self) -> None:
        if os.name == "nt":
            raise RuntimeError("keyboard_servo_control supports Linux terminals only.")

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok() and self.running:
                ready, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not ready:
                    continue

                key = sys.stdin.read(1)
                self.handle_key(key)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def handle_key(self, key: str) -> None:
        normalized = key.lower()

        if normalized == "q":
            self.get_logger().info("Quit requested from keyboard.")
            self.running = False
            self.stop_all()
            return

        if normalized == "\t":
            self.switch_arm()
            return

        if normalized == "r":
            self.reverse_direction()
            return

        if normalized == " ":
            self.stop_all()
            return

        command = self.key_to_joint_command(normalized)
        if command is None:
            return

        with self.lock:
            self.current_command = command
            self.last_key_time = time.monotonic()
            self.zero_sent = False

        self.get_logger().info(
            f"[{command.target}] {command.joint_name}: "
            f"velocity={command.velocity:.3f}",
            throttle_duration_sec=0.2,
        )

    def switch_arm(self) -> None:
        with self.lock:
            self.active_arm = "right" if self.active_arm == "left" else "left"
            self.current_command = None
            self.zero_sent = False

        self.stop_all()
        self.get_logger().info(f"Active arm switched to: {self.active_arm}")

    def reverse_direction(self) -> None:
        with self.lock:
            self.direction *= -1.0
            direction_text = "positive" if self.direction > 0.0 else "negative"

        self.get_logger().info(f"Arm jog direction switched to: {direction_text}")

    def key_to_joint_command(self, key: str) -> Optional[JointCommand]:
        if key == "w":
            return JointCommand("head", "neck_yaw_joint", self.head_joint_speed)
        if key == "s":
            return JointCommand("head", "neck_yaw_joint", -self.head_joint_speed)
        if key == "a":
            return JointCommand("head", "neck_pitch_joint", self.head_joint_speed)
        if key == "d":
            return JointCommand("head", "neck_pitch_joint", -self.head_joint_speed)

        if key not in {"1", "2", "3", "4", "5", "6"}:
            return None

        joint_index = int(key) - 1
        joint_name = self.joints_by_target[self.active_arm][joint_index]
        velocity = self.direction * self.arm_joint_speed
        return JointCommand(self.active_arm, joint_name, velocity)

    def publish_current_command(self) -> None:
        with self.lock:
            command = self.current_command
            elapsed = time.monotonic() - self.last_key_time if self.last_key_time else None

            if elapsed is not None and elapsed > self.command_timeout:
                self.current_command = None
                command = None
                if self.zero_sent:
                    return
                self.zero_sent = True

        if command is None:
            self.publish_zero_all()
            return

        self.publish_joint_jog(
            target=command.target,
            joint_names=[command.joint_name],
            velocities=[command.velocity],
        )

    def stop_all(self) -> None:
        with self.lock:
            self.current_command = None
            self.last_key_time = time.monotonic()
            self.zero_sent = True

        self.publish_zero_all()

    def publish_zero_all(self) -> None:
        for target, joint_names in self.joints_by_target.items():
            self.publish_joint_jog(
                target=target,
                joint_names=joint_names,
                velocities=[0.0] * len(joint_names),
            )

    def publish_joint_jog(
        self,
        target: str,
        joint_names: List[str],
        velocities: List[float],
    ) -> None:
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.command_frame
        msg.joint_names = joint_names
        msg.velocities = velocities
        msg.duration = self.command_timeout
        self._publishers[target].publish(msg)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[KeyboardServoControl] = None

    try:
        node = KeyboardServoControl()
        while rclpy.ok() and node.running:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop_all()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
