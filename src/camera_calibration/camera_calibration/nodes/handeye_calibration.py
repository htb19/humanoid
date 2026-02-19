#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from std_srvs.srv import Trigger
import tf2_ros
import cv2
import numpy as np
import threading
from tf_transformations import quaternion_matrix, quaternion_from_matrix

class HandEyeCalibration(Node):
    def __init__(self):
        super().__init__('handeye_calibration')

        # 声明参数
        self.camera_pose_topic = self.declare_parameter('camera_pose_topic', '/aruco_single/pose').value
        self.gripper_pose_topic = self.declare_parameter('gripper_pose_topic', '/end_effector_pose').value
        self.gripper_frame = self.declare_parameter('gripper_frame', 'arm_end_link').value   # 末端执行器坐标系
        self.camera_frame = self.declare_parameter('camera_frame', 'head_camera_optical_link').value  # 相机坐标系
        self.output_pose_topic = self.declare_parameter('output_pose_topic', '/cam2gripper_pose').value

        # 数据存储
        self.R_gripper2base_list = []   # 末端→基座的旋转矩阵
        self.t_gripper2base_list = []   # 末端→基座的平移向量
        self.R_target2cam_list = []     # 标定板→相机的旋转矩阵
        self.t_target2cam_list = []     # 标定板→相机的平移向量

        # 最新接收的位姿
        self.latest_camera_pose = None
        self.latest_gripper_pose = None
        self.lock = threading.Lock()

        # 订阅两个话题
        self.cam_sub = self.create_subscription(PoseStamped, self.camera_pose_topic, self.camera_callback, 10)
        self.gripper_sub = self.create_subscription(PoseStamped, self.gripper_pose_topic, self.gripper_callback, 10)

        # 发布结果（PoseStamped）
        self.pose_pub = self.create_publisher(PoseStamped, self.output_pose_topic, 10)

        # 静态 TF 广播器
        self.static_broadcaster = tf2_ros.StaticTransformBroadcaster(self)

        # 创建服务
        self.save_srv = self.create_service(Trigger, 'save_handeye_pair', self.save_callback)
        self.compute_srv = self.create_service(Trigger, 'compute_handeye', self.compute_callback)

        self.get_logger().info('手眼标定节点已启动，请使用服务 /save_handeye_pair 和 /compute_handeye 进行操作')

    def camera_callback(self, msg):
        """更新最新标定板位姿"""
        with self.lock:
            self.latest_camera_pose = msg

    def gripper_callback(self, msg):
        """更新最新末端位姿"""
        with self.lock:
            self.latest_gripper_pose = msg

    def pose_to_rt(self, pose_msg):
        """
        将 PoseStamped 转换为旋转矩阵 (3x3) 和平移向量 (3x1)
        """
        p = pose_msg.pose
        # 四元数转旋转矩阵
        quat = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        R = quaternion_matrix(quat)[:3, :3]
        t = np.array([[p.position.x], [p.position.y], [p.position.z]], dtype=np.float32)
        return R, t

    def save_current_pair(self):
        """保存当前最新的一对标定数据"""
        with self.lock:
            if self.latest_camera_pose is None or self.latest_gripper_pose is None:
                self.get_logger().warn('尚未接收到两个话题的数据，无法保存')
                return False

            # 提取两组数据
            R_tar2cam, t_tar2cam = self.pose_to_rt(self.latest_camera_pose)
            R_gri2base, t_gri2base = self.pose_to_rt(self.latest_gripper_pose)

        # 添加到列表
        self.R_target2cam_list.append(R_tar2cam)
        self.t_target2cam_list.append(t_tar2cam)
        self.R_gripper2base_list.append(R_gri2base)
        self.t_gripper2base_list.append(t_gri2base)

        self.get_logger().info(f'已保存第 {len(self.R_target2cam_list)} 组数据')
        return True

    def compute_and_publish(self):
        """计算手眼标定并发布结果"""
        n = len(self.R_target2cam_list)
        if n < 3:
            self.get_logger().warn(f'数据组数不足（当前{n}组），至少需要3组才能计算')
            return

        self.get_logger().info(f'开始使用 {n} 组数据进行手眼标定...')

        try:
            # 调用 OpenCV 的 calibrateHandEye
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                self.R_gripper2base_list, self.t_gripper2base_list,
                self.R_target2cam_list, self.t_target2cam_list,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
        except Exception as e:
            self.get_logger().error(f'标定失败: {e}')
            return

        # 转换为四元数
        T = np.eye(4)
        T[:3, :3] = R_cam2gripper
        T[:3, 3] = t_cam2gripper.flatten()
        quat = quaternion_from_matrix(T)

        self.get_logger().info('标定成功！')
        self.get_logger().info(f'平移: {t_cam2gripper.flatten()}')
        self.get_logger().info(f'四元数: {quat}')

        # 发布为 PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.gripper_frame
        pose_msg.pose.position.x = float(t_cam2gripper[0])
        pose_msg.pose.position.y = float(t_cam2gripper[1])
        pose_msg.pose.position.z = float(t_cam2gripper[2])
        pose_msg.pose.orientation.x = quat[0]
        pose_msg.pose.orientation.y = quat[1]
        pose_msg.pose.orientation.z = quat[2]
        pose_msg.pose.orientation.w = quat[3]
        self.pose_pub.publish(pose_msg)

        # 发布静态 TF (gripper_frame -> camera_frame)
        static_tf = TransformStamped()
        static_tf.header.stamp = self.get_clock().now().to_msg()
        static_tf.header.frame_id = self.gripper_frame
        static_tf.child_frame_id = self.camera_frame
        static_tf.transform.translation.x = float(t_cam2gripper[0])
        static_tf.transform.translation.y = float(t_cam2gripper[1])
        static_tf.transform.translation.z = float(t_cam2gripper[2])
        static_tf.transform.rotation.x = quat[0]
        static_tf.transform.rotation.y = quat[1]
        static_tf.transform.rotation.z = quat[2]
        static_tf.transform.rotation.w = quat[3]

        self.static_broadcaster.sendTransform(static_tf)
        self.get_logger().info(f'已发布静态 TF: {self.gripper_frame} -> {self.camera_frame}')

    # 服务回调
    def save_callback(self, request, response):
        success = self.save_current_pair()
        response.success = success
        response.message = f"已保存 {len(self.R_target2cam_list)} 组数据" if success else "保存失败"
        return response

    def compute_callback(self, request, response):
        self.compute_and_publish()
        response.success = True
        response.message = "计算完成并发布结果"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalibration()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
