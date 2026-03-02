#!/usr/bin/env python3
"""
Cartesian-level keyboard control for humanoid robot end-effectors
Keys:
  Left gripper (qweasdzxc): x+/-, y+/-, z+/-, roll+/-, pitch+/-, yaw+/-
  Right gripper (uiojklm,.): x+/-, y+/-, z+/-, roll+/-, pitch+/-, yaw+/-
  Neck (g,h): yaw, pitch
  ESC: quit
"""

import sys
import termios
import tty
import rclpy
from rclpy.node import Node
from robot_interfaces.msg import PoseCommand
from example_interfaces.msg import Float64MultiArray


class CartesianKeyboardControl(Node):
    def __init__(self):
        super().__init__('cartesian_keyboard_control')
        
        # Publishers
        self.left_pose_pub = self.create_publisher(PoseCommand, '/left_pose_command', 10)
        self.right_pose_pub = self.create_publisher(PoseCommand, '/right_pose_command', 10)
        self.neck_joint_pub = self.create_publisher(Float64MultiArray, '/neck_joint_command', 10)
        
        # Current poses (relative increments)
        self.position_step = 0.02  # meters
        self.orientation_step = 0.1  # radians
        self.neck_step = 0.1  # radians
        
        # Neck state
        self.neck_joints = [0.0, 0.0]
        
        self.get_logger().info('Cartesian Keyboard Control Started')
        self.print_instructions()
        
    def print_instructions(self):
        print("\n" + "="*60)
        print("CARTESIAN KEYBOARD CONTROL")
        print("="*60)
        print("Left Gripper (qweasdzxc):")
        print("  q/a = X+/-    w/s = Y+/-    e/d = Z+/-")
        print("  z/x = Roll+/- (not used)    c = Yaw+ (not used)")
        print("")
        print("Right Gripper (uiojklm,.):")
        print("  u/j = X+/-    i/k = Y+/-    o/l = Z+/-")
        print("  m/, = Roll+/- (not used)    . = Yaw+ (not used)")
        print("")
        print("Neck (g,h):")
        print("  g/G = Yaw+/-  h/H = Pitch+/-")
        print("")
        print("ESC or Ctrl+C: Quit")
        print("="*60 + "\n")
    
    def get_key(self):
        """Get single keypress from terminal"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
            if key == '\x1b':
                key += sys.stdin.read(2)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def publish_pose(self, publisher, x=0.0, y=0.0, z=0.0, roll=0.0, pitch=0.0, yaw=0.0):
        """Publish relative pose command"""
        msg = PoseCommand()
        msg.x = x
        msg.y = y
        msg.z = z
        msg.roll = roll
        msg.pitch = pitch
        msg.yaw = yaw
        msg.relative = True
        msg.cartesian_path = True
        publisher.publish(msg)
    
    def handle_key(self, key):
        """Process keypress and publish commands"""
        step = self.position_step
        rot_step = self.orientation_step
        
        # Left gripper control (qweasdzxc)
        if key == 'q':
            self.publish_pose(self.left_pose_pub, x=step)
            self.get_logger().info(f'Left: X+{step}')
        elif key == 'a':
            self.publish_pose(self.left_pose_pub, x=-step)
            self.get_logger().info(f'Left: X-{step}')
        elif key == 'w':
            self.publish_pose(self.left_pose_pub, y=step)
            self.get_logger().info(f'Left: Y+{step}')
        elif key == 's':
            self.publish_pose(self.left_pose_pub, y=-step)
            self.get_logger().info(f'Left: Y-{step}')
        elif key == 'e':
            self.publish_pose(self.left_pose_pub, z=step)
            self.get_logger().info(f'Left: Z+{step}')
        elif key == 'd':
            self.publish_pose(self.left_pose_pub, z=-step)
            self.get_logger().info(f'Left: Z-{step}')
        elif key == 'z':
            self.publish_pose(self.left_pose_pub, roll=rot_step)
            self.get_logger().info(f'Left: Roll+{rot_step}')
        elif key == 'x':
            self.publish_pose(self.left_pose_pub, roll=-rot_step)
            self.get_logger().info(f'Left: Roll-{rot_step}')
        elif key == 'c':
            self.publish_pose(self.left_pose_pub, yaw=rot_step)
            self.get_logger().info(f'Left: Yaw+{rot_step}')
            
        # Right gripper control (uiojklm,.)
        elif key == 'u':
            self.publish_pose(self.right_pose_pub, x=step)
            self.get_logger().info(f'Right: X+{step}')
        elif key == 'j':
            self.publish_pose(self.right_pose_pub, x=-step)
            self.get_logger().info(f'Right: X-{step}')
        elif key == 'i':
            self.publish_pose(self.right_pose_pub, y=step)
            self.get_logger().info(f'Right: Y+{step}')
        elif key == 'k':
            self.publish_pose(self.right_pose_pub, y=-step)
            self.get_logger().info(f'Right: Y-{step}')
        elif key == 'o':
            self.publish_pose(self.right_pose_pub, z=step)
            self.get_logger().info(f'Right: Z+{step}')
        elif key == 'l':
            self.publish_pose(self.right_pose_pub, z=-step)
            self.get_logger().info(f'Right: Z-{step}')
        elif key == 'm':
            self.publish_pose(self.right_pose_pub, roll=rot_step)
            self.get_logger().info(f'Right: Roll+{rot_step}')
        elif key == ',':
            self.publish_pose(self.right_pose_pub, roll=-rot_step)
            self.get_logger().info(f'Right: Roll-{rot_step}')
        elif key == '.':
            self.publish_pose(self.right_pose_pub, yaw=rot_step)
            self.get_logger().info(f'Right: Yaw+{rot_step}')
            
        # Neck control (g,h) — controller order: [pitch, yaw]
        elif key == 'g':
            self.neck_joints[1] += self.neck_step  # yaw is index 1
            msg = Float64MultiArray()
            msg.data = self.neck_joints.copy()
            self.neck_joint_pub.publish(msg)
            self.get_logger().info(f'Neck Yaw: {self.neck_joints[1]:.2f}')
        elif key == 'G':
            self.neck_joints[1] -= self.neck_step
            msg = Float64MultiArray()
            msg.data = self.neck_joints.copy()
            self.neck_joint_pub.publish(msg)
            self.get_logger().info(f'Neck Yaw: {self.neck_joints[1]:.2f}')
        elif key == 'h':
            self.neck_joints[0] += self.neck_step  # pitch is index 0
            msg = Float64MultiArray()
            msg.data = self.neck_joints.copy()
            self.neck_joint_pub.publish(msg)
            self.get_logger().info(f'Neck Pitch: {self.neck_joints[0]:.2f}')
        elif key == 'H':
            self.neck_joints[0] -= self.neck_step
            msg = Float64MultiArray()
            msg.data = self.neck_joints.copy()
            self.neck_joint_pub.publish(msg)
            self.get_logger().info(f'Neck Pitch: {self.neck_joints[0]:.2f}')
            
        # Quit
        elif key == '\x1b' or key == '\x03':
            return False
            
        return True
    
    def run(self):
        """Main control loop"""
        try:
            while rclpy.ok():
                key = self.get_key()
                if not self.handle_key(key):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.get_logger().info('Shutting down...')


def main(args=None):
    rclpy.init(args=args)
    node = CartesianKeyboardControl()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
