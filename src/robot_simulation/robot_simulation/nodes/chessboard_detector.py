#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
import tf2_ros
from tf_transformations import quaternion_from_matrix

class ChessboardDetector(Node):
    def __init__(self):
        super().__init__('chessboard_detector')

        # 声明并获取参数
        self.image_topic = self.declare_parameter('image_topic', '/camera/image_raw').value
        self.camera_info_topic = self.declare_parameter('camera_info_topic', '/camera/camera_info').value
        self.pattern_cols = self.declare_parameter('pattern_cols', 9).value
        self.pattern_rows = self.declare_parameter('pattern_rows', 6).value
        self.square_size = self.declare_parameter('square_size', 0.025).value
        self.camera_frame = self.declare_parameter('camera_frame', 'camera_optical_frame').value
        self.board_frame = self.declare_parameter('board_frame', 'chessboard').value
        self.pose_topic = self.declare_parameter('pose_topic', '/chessboard_pose').value

        self.pattern_size = (self.pattern_cols, self.pattern_rows)

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.have_camera_info = False

        # 生成世界坐标系下的3D点 (棋盘格坐标系: Z=0, 原点在第一个角点)
        self.objp = np.zeros((self.pattern_cols * self.pattern_rows, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.pattern_cols, 0:self.pattern_rows].T.reshape(-1, 2) * self.square_size

        # 订阅相机信息
        self.sub_camera_info = self.create_subscription(
            CameraInfo, self.camera_info_topic, self.camera_info_callback, 10
        )

        # 等待相机信息（使用简易循环，避免阻塞过久）
        self.get_logger().info("等待相机内参信息...")
        while not self.have_camera_info and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info("相机内参已获取，开始检测图像...")

        # 订阅图像话题
        self.sub_image = self.create_subscription(
            Image, self.image_topic, self.image_callback, 10
        )

        # 发布位姿
        self.pose_pub = self.create_publisher(PoseStamped, self.pose_topic, 10)

        # 发布 TF
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    def camera_info_callback(self, msg):
        """接收相机内参并保存"""
        self.camera_matrix = np.array(msg.k).reshape(3, 3)   # ROS2 中为 msg.k
        self.dist_coeffs = np.array(msg.d)                   # ROS2 中为 msg.d
        self.have_camera_info = True
        self.get_logger().info(f"收到相机内参:\n{self.camera_matrix}")

    def image_callback(self, msg):
        """处理图像：检测棋盘格并发布位姿和 TF"""
        if not self.have_camera_info:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, self.pattern_size, None)
        self.get_logger().info(f"findChessboardCorners 返回: {ret}")
        if ret:
            self.get_logger().info(f"检测到 {len(corners)} 个角点")
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            ret, rvec, tvec = cv2.solvePnP(self.objp, corners2, self.camera_matrix, self.dist_coeffs)

            if ret:
                # 旋转向量 -> 旋转矩阵 -> 齐次变换矩阵 -> 四元数
                R, _ = cv2.Rodrigues(rvec)
                T_board_to_cam = np.eye(4)
                T_board_to_cam[:3, :3] = R
                T_board_to_cam[:3, 3] = tvec.flatten()
                quaternion = quaternion_from_matrix(T_board_to_cam)

                # 发布 PoseStamped
                pose_msg = PoseStamped()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = self.camera_frame
                pose_msg.pose.position.x = float(tvec[0])
                pose_msg.pose.position.y = float(tvec[1])
                pose_msg.pose.position.z = float(tvec[2])
                pose_msg.pose.orientation.x = quaternion[0]
                pose_msg.pose.orientation.y = quaternion[1]
                pose_msg.pose.orientation.z = quaternion[2]
                pose_msg.pose.orientation.w = quaternion[3]
                self.pose_pub.publish(pose_msg)

                # 发布 TF 变换（camera_frame -> board_frame）
                t = TransformStamped()
                t.header.stamp = self.get_clock().now().to_msg()
                t.header.frame_id = self.camera_frame
                t.child_frame_id = self.board_frame
                t.transform.translation.x = float(tvec[0])
                t.transform.translation.y = float(tvec[1])
                t.transform.translation.z = float(tvec[2])
                t.transform.rotation.x = quaternion[0]
                t.transform.rotation.y = quaternion[1]
                t.transform.rotation.z = quaternion[2]
                t.transform.rotation.w = quaternion[3]
                self.tf_broadcaster.sendTransform(t)

                # 可选：在图像上绘制角点和坐标轴（调试用）
                cv2.drawChessboardCorners(cv_image, self.pattern_size, corners2, ret)
                axis_points = np.float32([[0.05, 0, 0], [0, 0.05, 0], [0, 0, -0.05]]).reshape(-1, 3)
                img_points, _ = cv2.projectPoints(axis_points, rvec, tvec, self.camera_matrix, self.dist_coeffs)
                origin = tuple(corners2[0].ravel().astype(int))
                cv_image = cv2.line(cv_image, origin, tuple(img_points[0].ravel().astype(int)), (0, 0, 255), 3)
                cv_image = cv2.line(cv_image, origin, tuple(img_points[1].ravel().astype(int)), (0, 255, 0), 3)
                cv_image = cv2.line(cv_image, origin, tuple(img_points[2].ravel().astype(int)), (255, 0, 0), 3)
                # 如需显示图像，可取消下面注释（但可能会干扰 ROS 主循环）
                cv2.imshow("Chessboard Detection", cv_image)
                cv2.waitKey(1)
            else:
                self.get_logger().warn("solvePnP 失败")
        else:
            self.get_logger().debug("未检测到棋盘格")

def main(args=None):
    rclpy.init(args=args)
    node = ChessboardDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
