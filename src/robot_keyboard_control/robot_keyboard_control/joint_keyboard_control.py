#!/usr/bin/env python3
"""
Joint-level keyboard control for humanoid robot
Keys:
  Left arm (1-7): base_pitch, shoulder_yaw, shoulder_roll, elbow_pitch, wrist_pitch, wrist_yaw, gripper
  Right arm (q-u): base_pitch, shoulder_yaw, shoulder_roll, elbow_pitch, wrist_pitch, wrist_yaw, gripper
  Neck (a,s): yaw, pitch
  ESC: quit
"""

import sys
import termios
import tty
import rclpy
from rclpy.node import Node
from example_interfaces.msg import Float64MultiArray, Bool


class JointKeyboardControl(Node):
    def __init__(self):
        super().__init__('joint_keyboard_control')
        
        # Publishers
        self.left_joint_pub = self.create_publisher(Float64MultiArray, '/left_joint_command', 10)
        self.right_joint_pub = self.create_publisher(Float64MultiArray, '/right_joint_command', 10)
        self.neck_joint_pub = self.create_publisher(Float64MultiArray, '/neck_joint_command', 10)
        self.left_gripper_pub = self.create_publisher(Bool, '/open_left_gripper', 10)
        self.right_gripper_pub = self.create_publisher(Bool, '/open_right_gripper', 10)
        
        # Joint states (6 DOF for arms, 2 DOF for neck)
        self.left_joints = [0.0] * 6
        self.right_joints = [0.0] * 6
        self.neck_joints = [0.0, 0.0]
        
        # Control parameters
        self.joint_step = 0.1  # radians
        
        self.get_logger().info('Joint Keyboard Control Started')
        self.print_instructions()
        
    def print_instructions(self):
        print("\n" + "="*60)
        print("JOINT KEYBOARD CONTROL")
        print("="*60)
        print("Left Arm (1-6):  1=base_pitch 2=shoulder_roll 3=shoulder_yaw")
        print("                 4=elbow_pitch 5=wrist_pitch 6=wrist_yaw")
        print("Left Gripper:    7=toggle open/close")
        print("")
        print("Right Arm (q-y): q=base_pitch w=shoulder_roll e=shoulder_yaw")
        print("                 r=elbow_pitch t=wrist_pitch y=wrist_yaw")
        print("Right Gripper:   u=toggle open/close")
        print("")
        print("Neck (a,s):      a=yaw s=pitch")
        print("")
        print("Modifiers:       SHIFT=decrease, normal=increase")
        print("ESC or Ctrl+C:   Quit")
        print("="*60 + "\n")
    
    def get_key(self):
        """Get single keypress from terminal"""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)
            # Handle escape sequences
            if key == '\x1b':
                key += sys.stdin.read(2)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return key
    
    def handle_key(self, key):
        """Process keypress and publish commands"""
        # Left arm joints (1-6)
        left_keys = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5,
                     '!': 0, '@': 1, '#': 2, '$': 3, '%': 4, '^': 5}
        # Right arm joints (q-y)
        right_keys = {'q': 0, 'w': 1, 'e': 2, 'r': 3, 't': 4, 'y': 5,
                      'Q': 0, 'W': 1, 'E': 2, 'R': 3, 'T': 4, 'Y': 5}
        # Neck joints (a=yaw[1], s=pitch[0]) — controller order: [pitch, yaw]
        neck_keys = {'a': 1, 's': 0, 'A': 1, 'S': 0}
        
        # Determine direction (uppercase/shift = decrease)
        direction = -1 if key.isupper() or key in '!@#$%^' else 1
        
        # Left arm control
        if key.lower() in left_keys:
            joint_idx = left_keys[key.lower() if key.islower() else key]
            self.left_joints[joint_idx] += direction * self.joint_step
            msg = Float64MultiArray()
            msg.data = self.left_joints
            self.left_joint_pub.publish(msg)
            self.get_logger().info(f'Left joint {joint_idx}: {self.left_joints[joint_idx]:.2f}')
            
        # Right arm control
        elif key.lower() in right_keys:
            joint_idx = right_keys[key]
            self.right_joints[joint_idx] += direction * self.joint_step
            msg = Float64MultiArray()
            msg.data = self.right_joints
            self.right_joint_pub.publish(msg)
            self.get_logger().info(f'Right joint {joint_idx}: {self.right_joints[joint_idx]:.2f}')
            
        # Neck control
        elif key.lower() in neck_keys:
            joint_idx = neck_keys[key]
            self.neck_joints[joint_idx] += direction * self.joint_step
            msg = Float64MultiArray()
            msg.data = self.neck_joints
            self.neck_joint_pub.publish(msg)
            self.get_logger().info(f'Neck joint {joint_idx}: {self.neck_joints[joint_idx]:.2f}')
            
        # Gripper control
        elif key == '7':
            msg = Bool()
            msg.data = True
            self.left_gripper_pub.publish(msg)
            self.get_logger().info('Left gripper: OPEN')
        elif key == '&':
            msg = Bool()
            msg.data = False
            self.left_gripper_pub.publish(msg)
            self.get_logger().info('Left gripper: CLOSE')
        elif key == 'u':
            msg = Bool()
            msg.data = True
            self.right_gripper_pub.publish(msg)
            self.get_logger().info('Right gripper: OPEN')
        elif key == 'U':
            msg = Bool()
            msg.data = False
            self.right_gripper_pub.publish(msg)
            self.get_logger().info('Right gripper: CLOSE')
            
        # Quit
        elif key == '\x1b' or key == '\x03':  # ESC or Ctrl+C
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
    node = JointKeyboardControl()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
