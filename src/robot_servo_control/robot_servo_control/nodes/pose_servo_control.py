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
from typing import Dict, Optional

import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.publisher import Publisher
from std_srvs.srv import Trigger


@dataclass(frozen=True)
class TwistCommand:
    target: str
    component: str
    axis: str
    velocity: float


class PoseServoControl(Node):
    def __init__(self) -> None:
        super().__init__("pose_servo_control")

        self.declare_parameter(
            "left_twist_topic",
            "/left_servo/moveit_servo/delta_twist_cmds",
        )
        self.declare_parameter(
            "right_twist_topic",
            "/right_servo/moveit_servo/delta_twist_cmds",
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
        self.declare_parameter("linear_speed", 0.1)
        self.declare_parameter("angular_speed", 0.3)
        self.declare_parameter("publish_rate", 20.0)
        self.declare_parameter("command_timeout", 0.2)
        self.declare_parameter("wait_service_timeout_sec", 30.0)

        self.command_frame = self.get_string_param("command_frame")
        self.linear_speed = self.get_double_param("linear_speed")
        self.angular_speed = self.get_double_param("angular_speed")
        self.publish_rate = self.get_double_param("publish_rate")
        self.command_timeout = self.get_double_param("command_timeout")
        self.wait_service_timeout_sec = self.get_double_param(
            "wait_service_timeout_sec"
        )

        self._publishers: Dict[str, Publisher] = {
            "left": self.create_publisher(
                TwistStamped,
                self.get_string_param("left_twist_topic"),
                10,
            ),
            "right": self.create_publisher(
                TwistStamped,
                self.get_string_param("right_twist_topic"),
                10,
            ),
        }
        self.start_services = {
            "left": self.get_string_param("left_start_service"),
            "right": self.get_string_param("right_start_service"),
        }

        self.active_arm = "left"
        self.current_command: Optional[TwistCommand] = None
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
            "Pose servo control ready.\n"
            "Translation control:\n"
            "  q/a: x axis positive/negative\n"
            "  w/s: y axis positive/negative\n"
            "  e/d: z axis positive/negative\n"
            "Rotation control:\n"
            "  r/f: x axis positive/negative\n"
            "  t/g: y axis positive/negative\n"
            "  y/h: z axis positive/negative\n"
            "Other:\n"
            "  Tab: switch selected arm between left/right\n"
            "  space: stop all\n"
            "  Esc or Ctrl-C: quit node\n"
        )
        self.get_logger().info(help_text)

    def keyboard_loop(self) -> None:
        if os.name == "nt":
            raise RuntimeError("pose_servo_control supports Linux terminals only.")

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

        if normalized == "\x1b":
            self.get_logger().info("Quit requested from keyboard.")
            self.running = False
            self.stop_all()
            return

        if normalized == "\t":
            self.switch_arm()
            return

        if normalized == " ":
            self.stop_all()
            return

        command = self.key_to_twist_command(normalized)
        if command is None:
            return

        with self.lock:
            self.current_command = command
            self.last_key_time = time.monotonic()
            self.zero_sent = False

        self.get_logger().info(
            f"[{command.target}] {command.component}.{command.axis}: "
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

    def key_to_twist_command(self, key: str) -> Optional[TwistCommand]:
        linear_keys = {
            "q": ("x", self.linear_speed),
            "a": ("x", -self.linear_speed),
            "w": ("y", self.linear_speed),
            "s": ("y", -self.linear_speed),
            "e": ("z", self.linear_speed),
            "d": ("z", -self.linear_speed),
        }
        angular_keys = {
            "r": ("x", self.angular_speed),
            "f": ("x", -self.angular_speed),
            "t": ("y", self.angular_speed),
            "g": ("y", -self.angular_speed),
            "y": ("z", self.angular_speed),
            "h": ("z", -self.angular_speed),
        }

        if key in linear_keys:
            axis, velocity = linear_keys[key]
            return TwistCommand(self.active_arm, "linear", axis, velocity)

        if key in angular_keys:
            axis, velocity = angular_keys[key]
            return TwistCommand(self.active_arm, "angular", axis, velocity)

        return None

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

        self.publish_twist(command)

    def stop_all(self) -> None:
        with self.lock:
            self.current_command = None
            self.last_key_time = time.monotonic()
            self.zero_sent = True

        self.publish_zero_all()

    def publish_zero_all(self) -> None:
        for target in self._publishers:
            self.publish_zero(target)

    def publish_zero(self, target: str) -> None:
        self._publishers[target].publish(self.build_twist_msg())

    def publish_twist(self, command: TwistCommand) -> None:
        msg = self.build_twist_msg()
        vector = getattr(msg.twist, command.component)
        setattr(vector, command.axis, command.velocity)
        self._publishers[command.target].publish(msg)

    def build_twist_msg(self) -> TwistStamped:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.command_frame
        return msg


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[PoseServoControl] = None

    try:
        node = PoseServoControl()
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
