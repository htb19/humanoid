#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.client import Client
from rclpy.task import Future
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    HistoryPolicy,
    DurabilityPolicy,
)
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger


class TwistStampBridge(Node):
    def __init__(self) -> None:
        super().__init__("twist_stamp_bridge")

        # =========================
        # 参数声明
        # =========================
        self.declare_parameter("left_input_topic", "/unity/left_twist_raw")
        self.declare_parameter("right_input_topic", "/unity/right_twist_raw")

        self.declare_parameter(
            "left_output_topic",
            "/left_servo/moveit_servo/delta_twist_cmds",
        )
        self.declare_parameter(
            "right_output_topic",
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

        self.declare_parameter("enable_left", True)
        self.declare_parameter("enable_right", True)

        self.declare_parameter("reliability", "reliable")
        self.declare_parameter("depth", 10)
        self.declare_parameter("wait_service_timeout_sec", 100.0)
        self.declare_parameter("log_interval_sec", 2.0)
        self.declare_parameter("stale_timeout_sec", 0.2)
        self.declare_parameter("expected_frame_id", "base_link")
        self.declare_parameter("enforce_frame_id", True)
        self.declare_parameter("max_linear_speed", 1.2)
        self.declare_parameter("max_angular_speed", 2.5)
        self.declare_parameter("zero_publish_repeats", 3)

        # =========================
        # 参数读取（类型安全）
        # =========================
        self.left_input_topic: str = (
            self.get_parameter("left_input_topic")
            .get_parameter_value()
            .string_value
        )
        self.right_input_topic: str = (
            self.get_parameter("right_input_topic")
            .get_parameter_value()
            .string_value
        )

        self.left_output_topic: str = (
            self.get_parameter("left_output_topic")
            .get_parameter_value()
            .string_value
        )
        self.right_output_topic: str = (
            self.get_parameter("right_output_topic")
            .get_parameter_value()
            .string_value
        )

        self.left_start_service: str = (
            self.get_parameter("left_start_service")
            .get_parameter_value()
            .string_value
        )
        self.right_start_service: str = (
            self.get_parameter("right_start_service")
            .get_parameter_value()
            .string_value
        )

        self.enable_left: bool = (
            self.get_parameter("enable_left")
            .get_parameter_value()
            .bool_value
        )
        self.enable_right: bool = (
            self.get_parameter("enable_right")
            .get_parameter_value()
            .bool_value
        )

        reliability_str: str = (
            self.get_parameter("reliability")
            .get_parameter_value()
            .string_value
            .lower()
        )

        depth: int = (
            self.get_parameter("depth")
            .get_parameter_value()
            .integer_value
        )

        self.wait_service_timeout_sec: float = (
            self.get_parameter("wait_service_timeout_sec")
            .get_parameter_value()
            .double_value
        )

        self.log_interval_sec: float = (
            self.get_parameter("log_interval_sec")
            .get_parameter_value()
            .double_value
        )
        self.stale_timeout_sec: float = (
            self.get_parameter("stale_timeout_sec")
            .get_parameter_value()
            .double_value
        )
        self.expected_frame_id: str = (
            self.get_parameter("expected_frame_id")
            .get_parameter_value()
            .string_value
        )
        self.enforce_frame_id: bool = (
            self.get_parameter("enforce_frame_id")
            .get_parameter_value()
            .bool_value
        )
        self.max_linear_speed: float = (
            self.get_parameter("max_linear_speed")
            .get_parameter_value()
            .double_value
        )
        self.max_angular_speed: float = (
            self.get_parameter("max_angular_speed")
            .get_parameter_value()
            .double_value
        )
        self.zero_publish_repeats: int = (
            self.get_parameter("zero_publish_repeats")
            .get_parameter_value()
            .integer_value
        )

        # =========================
        # QoS 配置
        # =========================
        reliability: ReliabilityPolicy
        if reliability_str == "reliable":
            reliability = ReliabilityPolicy.RELIABLE
        else:
            reliability = ReliabilityPolicy.BEST_EFFORT

        self.qos: QoSProfile = QoSProfile(
            reliability=reliability,
            history=HistoryPolicy.KEEP_LAST,
            depth=depth,
            durability=DurabilityPolicy.VOLATILE,
        )

        # =========================
        # 成员初始化
        # =========================
        self.left_pub = None
        self.right_pub = None
        self.left_sub = None
        self.right_sub = None
        self.timer = None

        self.left_count: int = 0
        self.right_count: int = 0
        self.left_reject_count: int = 0
        self.right_reject_count: int = 0
        self.left_zero_count: int = 0
        self.right_zero_count: int = 0
        self.left_last_recv_time = None
        self.right_last_recv_time = None
        self.left_last_forward_time = None
        self.right_last_forward_time = None
        self.left_last_frame_id: str = ""
        self.right_last_frame_id: str = ""
        self.left_last_linear_norm: float = 0.0
        self.right_last_linear_norm: float = 0.0
        self.left_last_angular_norm: float = 0.0
        self.right_last_angular_norm: float = 0.0
        self.left_zero_remaining: int = 0
        self.right_zero_remaining: int = 0
        self.left_stale_active: bool = False
        self.right_stale_active: bool = False

        self.get_logger().info("TwistStampBridge starting...")
        self.get_logger().info(
            f"QoS: reliability={reliability_str}, depth={depth}, "
            f"history=keep_last, durability=volatile"
        )
        self.get_logger().info(
            "Guard rails: "
            f"stale_timeout={self.stale_timeout_sec:.3f}s, "
            f"expected_frame_id='{self.expected_frame_id}', "
            f"enforce_frame_id={self.enforce_frame_id}, "
            f"max_linear_speed={self.max_linear_speed:.3f}, "
            f"max_angular_speed={self.max_angular_speed:.3f}, "
            f"zero_publish_repeats={self.zero_publish_repeats}"
        )

        # =========================
        # 先创建 publisher
        # =========================
        if self.enable_left:
            self.left_pub = self.create_publisher(
                TwistStamped,
                self.left_output_topic,
                self.qos,
            )

        if self.enable_right:
            self.right_pub = self.create_publisher(
                TwistStamped,
                self.right_output_topic,
                self.qos,
            )

        # =========================
        # 先激活 Servo，再开始订阅 Unity
        # =========================
        if self.enable_left:
            ok_left: bool = self.start_servo(self.left_start_service, "left")
            if not ok_left:
                raise RuntimeError(
                    f"Failed to start left servo: {self.left_start_service}"
                )

        if self.enable_right:
            ok_right: bool = self.start_servo(self.right_start_service, "right")
            if not ok_right:
                raise RuntimeError(
                    f"Failed to start right servo: {self.right_start_service}"
                )

        # =========================
        # Servo 激活成功后再创建订阅
        # =========================
        if self.enable_left:
            self.left_sub = self.create_subscription(
                TwistStamped,
                self.left_input_topic,
                self.left_cb,
                self.qos,
            )

        if self.enable_right:
            self.right_sub = self.create_subscription(
                TwistStamped,
                self.right_input_topic,
                self.right_cb,
                self.qos,
            )

        self.timer = self.create_timer(
            self.log_interval_sec,
            self.log_status,
        )

        self.get_logger().info("TwistStampBridge ready.")

        if self.enable_left:
            self.get_logger().info(
                f"Left : {self.left_input_topic} -> {self.left_output_topic}"
            )

        if self.enable_right:
            self.get_logger().info(
                f"Right: {self.right_input_topic} -> {self.right_output_topic}"
            )

    def start_servo(self, service_name: str, tag: str) -> bool:
        client: Client = self.create_client(Trigger, service_name)

        self.get_logger().info(f"[{tag}] waiting for service: {service_name}")
        available: bool = client.wait_for_service(
            timeout_sec=self.wait_service_timeout_sec
        )
        if not available:
            self.get_logger().error(
                f"[{tag}] service not available: {service_name}"
            )
            return False

        req: Trigger.Request = Trigger.Request()
        future: Future = client.call_async(req)

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=self.wait_service_timeout_sec,
        )

        if not future.done():
            self.get_logger().error(
                f"[{tag}] start_servo call timed out: {service_name}"
            )
            return False

        result: Optional[Trigger.Response] = future.result()
        if result is None:
            self.get_logger().error(
                f"[{tag}] start_servo call returned None: {service_name}"
            )
            return False

        if result.success:
            self.get_logger().info(
                f"[{tag}] start_servo success: {result.message}"
            )
            return True

        self.get_logger().error(
            f"[{tag}] start_servo failed: {result.message}"
        )
        return False

    def left_cb(self, msg: TwistStamped) -> None:
        if self.left_pub is None:
            return

        out = self.validate_and_build_output(msg, "left")
        if out is None:
            self.left_reject_count += 1
            return

        self.left_pub.publish(out)
        self.left_count += 1
        now = self.get_clock().now()
        self.left_last_recv_time = now
        self.left_last_forward_time = now
        self.left_last_frame_id = out.header.frame_id
        self.left_last_linear_norm = self.vector_norm(out.twist.linear)
        self.left_last_angular_norm = self.vector_norm(out.twist.angular)
        self.left_zero_remaining = self.zero_publish_repeats
        self.left_stale_active = False

    def right_cb(self, msg: TwistStamped) -> None:
        if self.right_pub is None:
            return

        out = self.validate_and_build_output(msg, "right")
        if out is None:
            self.right_reject_count += 1
            return

        self.right_pub.publish(out)
        self.right_count += 1
        now = self.get_clock().now()
        self.right_last_recv_time = now
        self.right_last_forward_time = now
        self.right_last_frame_id = out.header.frame_id
        self.right_last_linear_norm = self.vector_norm(out.twist.linear)
        self.right_last_angular_norm = self.vector_norm(out.twist.angular)
        self.right_zero_remaining = self.zero_publish_repeats
        self.right_stale_active = False

    def log_status(self) -> None:
        self.check_stale_inputs()

        self.get_logger().info(
            "bridge alive | "
            f"left forwarded={self.left_count} rejected={self.left_reject_count} "
            f"zero={self.left_zero_count} frame={self.left_last_frame_id or '-'} "
            f"lin={self.left_last_linear_norm:.3f} ang={self.left_last_angular_norm:.3f} | "
            f"right forwarded={self.right_count} rejected={self.right_reject_count} "
            f"zero={self.right_zero_count} frame={self.right_last_frame_id or '-'} "
            f"lin={self.right_last_linear_norm:.3f} ang={self.right_last_angular_norm:.3f}"
        )

    def check_stale_inputs(self) -> None:
        now = self.get_clock().now()

        if self.enable_left and self.left_pub is not None:
            self.handle_stale_hand(
                tag="left",
                publisher=self.left_pub,
                last_recv_time=self.left_last_recv_time,
                now=now,
            )

        if self.enable_right and self.right_pub is not None:
            self.handle_stale_hand(
                tag="right",
                publisher=self.right_pub,
                last_recv_time=self.right_last_recv_time,
                now=now,
            )

    def handle_stale_hand(self, tag: str, publisher, last_recv_time, now) -> None:
        if last_recv_time is None:
            return

        elapsed = (now - last_recv_time).nanoseconds * 1e-9
        if elapsed <= self.stale_timeout_sec:
            return

        stale_flag_attr = f"{tag}_stale_active"
        zero_remaining_attr = f"{tag}_zero_remaining"
        last_forward_attr = f"{tag}_last_forward_time"
        zero_count_attr = f"{tag}_zero_count"

        if not getattr(self, stale_flag_attr):
            self.get_logger().warn(
                f"[{tag}] input stale for {elapsed:.3f}s, publishing zero twist",
                throttle_duration_sec=1.0,
            )
            setattr(self, stale_flag_attr, True)

        zero_remaining = getattr(self, zero_remaining_attr)
        if zero_remaining <= 0:
            return

        publisher.publish(self.build_zero_msg())
        setattr(self, zero_remaining_attr, zero_remaining - 1)
        setattr(self, last_forward_attr, now)
        setattr(self, zero_count_attr, getattr(self, zero_count_attr) + 1)

    def validate_and_build_output(
        self,
        msg: TwistStamped,
        tag: str,
    ) -> Optional[TwistStamped]:
        input_frame_id = msg.header.frame_id.strip()
        frame_id = self.expected_frame_id if self.enforce_frame_id else input_frame_id

        if not self.enforce_frame_id and input_frame_id != self.expected_frame_id:
            self.get_logger().warn(
                f"[{tag}] unexpected frame_id '{input_frame_id}', "
                f"expected '{self.expected_frame_id}'",
                throttle_duration_sec=1.0,
            )
            return None

        if not self.twist_is_finite(msg):
            self.get_logger().warn(
                f"[{tag}] rejected non-finite twist input",
                throttle_duration_sec=1.0,
            )
            return None

        linear_norm = self.vector_norm(msg.twist.linear)
        angular_norm = self.vector_norm(msg.twist.angular)
        if linear_norm > self.max_linear_speed:
            self.get_logger().warn(
                f"[{tag}] rejected linear speed {linear_norm:.3f} > "
                f"{self.max_linear_speed:.3f}",
                throttle_duration_sec=1.0,
            )
            return None

        if angular_norm > self.max_angular_speed:
            self.get_logger().warn(
                f"[{tag}] rejected angular speed {angular_norm:.3f} > "
                f"{self.max_angular_speed:.3f}",
                throttle_duration_sec=1.0,
            )
            return None

        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = frame_id
        out.twist = msg.twist
        return out

    def build_zero_msg(self) -> TwistStamped:
        out = TwistStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.expected_frame_id
        return out

    @staticmethod
    def twist_is_finite(msg: TwistStamped) -> bool:
        values = (
            msg.twist.linear.x,
            msg.twist.linear.y,
            msg.twist.linear.z,
            msg.twist.angular.x,
            msg.twist.angular.y,
            msg.twist.angular.z,
        )
        return all(math.isfinite(v) for v in values)

    @staticmethod
    def vector_norm(vec) -> float:
        return math.sqrt(vec.x * vec.x + vec.y * vec.y + vec.z * vec.z)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node: Optional[TwistStampBridge] = None

    try:
        node = TwistStampBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[twist_stamp_bridge] fatal: {exc}")
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
