#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
from tf2_ros import TransformException

class EndEffectorPoseTF(Node):
    def __init__(self):
        super().__init__('end_effector_pose_tf')

        # 声明参数
        self.source_frame = self.declare_parameter('source_frame', 'base_link').value
        self.target_frame = self.declare_parameter('target_frame', 'arm_end_link').value
        self.pose_topic = self.declare_parameter('pose_topic', '/end_effector_pose').value
        self.publish_rate = self.declare_parameter('publish_rate', 20.0).value

        # TF2 缓冲区和监听器
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 发布器
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)

        # 定时器，按指定频率发布
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        self.get_logger().info(
            f'节点已启动，监听 {self.source_frame} -> {self.target_frame}，发布到 {self.pose_topic}'
        )

    def timer_callback(self):
        try:
            # 查找当前最新的变换
            trans: TransformStamped = self.tf_buffer.lookup_transform(
                self.source_frame,
                self.target_frame,
                rclpy.time.Time()  # 获取最新变换
            )
        except TransformException as ex:
            self.get_logger().debug(f'变换不可用: {ex}')
            return

        # 构建 PoseStamped 消息
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.source_frame
        pose_msg.pose.position.x = trans.transform.translation.x
        pose_msg.pose.position.y = trans.transform.translation.y
        pose_msg.pose.position.z = trans.transform.translation.z
        pose_msg.pose.orientation = trans.transform.rotation

        # 发布
        self.pose_pub.publish(pose_msg)
        self.get_logger().debug('已发布位姿', throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = EndEffectorPoseTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
