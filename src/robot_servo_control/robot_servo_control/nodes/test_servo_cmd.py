#!/usr/bin/env python3
"""
MoveIt Servo 双臂测试脚本
同时测试 left_servo 和 right_servo
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from control_msgs.msg import JointJog
from std_srvs.srv import Trigger


class DualServoTestNode(Node):
    def __init__(self):
        super().__init__("dual_servo_test_node")

        # ── 右臂发布器 ───────────────────────────────────────
        self.right_twist_pub = self.create_publisher(
            TwistStamped,
            "/right_servo/moveit_servo/delta_twist_cmds",
            10,
        )
        self.right_joint_pub = self.create_publisher(
            JointJog,
            "/right_servo/moveit_servo/delta_joint_cmds",
            10,
        )

        # ── 左臂发布器 ───────────────────────────────────────
        self.left_twist_pub = self.create_publisher(
            TwistStamped,
            "/left_servo/moveit_servo/delta_twist_cmds",
            10,
        )
        self.left_joint_pub = self.create_publisher(
            JointJog,
            "/left_servo/moveit_servo/delta_joint_cmds",
            10,
        )

        # ── 右臂激活服务 ─────────────────────────────────────
        self.right_start_client = self.create_client(
            Trigger,
            "/right_servo/moveit_servo/start_servo",
        )

        # ── 左臂激活服务 ─────────────────────────────────────
        self.left_start_client = self.create_client(
            Trigger,
            "/left_servo/moveit_servo/start_servo",
        )

        # ── 定时器 10Hz ──────────────────────────────────────
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.count = 0

        # 激活双臂
        self.activate_servo("right", self.right_start_client)
        self.activate_servo("left", self.left_start_client)

    # ── 激活服务 ─────────────────────────────────────────────
    def activate_servo(self, arm_name, client):
        self.get_logger().info(f"等待 {arm_name}_servo start 服务...")
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"{arm_name}_servo start 服务不可用！")
            return
        future = client.call_async(Trigger.Request())
        future.add_done_callback(
            lambda f: self.activate_callback(f, arm_name)
        )

    def activate_callback(self, future, arm_name):
        result = future.result()
        if result.success:
            self.get_logger().info(f"{arm_name}_servo 已激活")
        else:
            self.get_logger().warn(f"{arm_name}_servo 激活失败: {result.message}")

    # ── 笛卡尔指令 ───────────────────────────────────────────
    def publish_twist(self, publisher, linear_x, linear_y, linear_z, arm_name):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.twist.linear.x = linear_x
        msg.twist.linear.y = linear_y
        msg.twist.linear.z = linear_z
        publisher.publish(msg)
        self.get_logger().info(
            f"[{arm_name}][Twist] linear.x={linear_x:.3f}",
            throttle_duration_sec=1.0,
        )

    # ── 关节指令 ─────────────────────────────────────────────
    def publish_joint_jog(self, publisher, joint_name, velocity, arm_name):
        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.joint_names = [joint_name]
        msg.velocities = [velocity]
        publisher.publish(msg)
        self.get_logger().info(
            f"[{arm_name}][Joint] {joint_name}={velocity:.3f} rad/s",
            throttle_duration_sec=1.0,
        )

    # ── 停止 ─────────────────────────────────────────────────
    def stop_all(self):
        for pub in [self.right_twist_pub, self.left_twist_pub]:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            pub.publish(msg)
        self.get_logger().info("⏹️  双臂已停止")

    # ── 定时回调 ─────────────────────────────────────────────
    def timer_callback(self):
        self.count += 1

        if self.count <= 50:
            # 前 5 秒：双臂同时笛卡尔测试
            self.publish_twist(self.right_twist_pub, 0.0, 0.00, 0.1, "right")
            self.publish_twist(self.left_twist_pub, -0.01, 0.0, 0.0, "left")

        # elif self.count <= 100:
            # 5~10 秒：双臂同时关节测试
            # 修改为你机器人实际关节名
            # self.publish_joint_jog(
            #     self.right_joint_pub,
            #     "right_base_pitch_joint",
            #     0.2,
            #     "right",
            # )
            # self.publish_joint_jog(
            #     self.left_joint_pub,
            #     "left_base_pitch_joint",
            #     0.2,
            #     "left",
            # )

        else:
            self.stop_all()
            self.get_logger().info("✅ 测试完成")
            self.timer.cancel()


def main():
    rclpy.init()
    node = DualServoTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("用户中断，停止双臂...")
        node.stop_all()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()