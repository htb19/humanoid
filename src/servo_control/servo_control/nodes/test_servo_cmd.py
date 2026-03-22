#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped

class CmdPublisher(Node):
    def __init__(self):
        super().__init__('cmd_pub')
        self.pub = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        self.timer = self.create_timer(0.05, self.publish_cmd)  # 20 Hz

    def publish_cmd(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.y = 0.4   # 沿 Y 轴移动
        msg.twist.linear.z = 0.4   # 沿 Z 轴移动
        self.pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = CmdPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
