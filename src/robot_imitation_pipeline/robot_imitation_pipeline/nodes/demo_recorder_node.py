#!/usr/bin/env python3
from pathlib import Path
import time
from typing import Dict, List, Optional

import numpy as np
import rclpy
from example_interfaces.msg import Bool, Float64MultiArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import SetBool, Trigger

from robot_imitation_pipeline.io_utils import ACTION_COMPONENTS, next_episode_dir, now_to_float, stamp_to_float, write_json

try:
    import cv2
    from cv_bridge import CvBridge
except ImportError:
    cv2 = None
    CvBridge = None


class DemoRecorder(Node):
    def __init__(self) -> None:
        super().__init__("demo_recorder")
        self._declare_parameters()
        self._load_parameters()

        self.bridge = CvBridge() if CvBridge is not None else None
        self.recording = False
        self.episode_dir: Optional[Path] = None
        self.episode_start_wall: Optional[float] = None
        self.episode_start_ros: Optional[float] = None

        self.latest_joint_state: Optional[JointState] = None
        self.latest_joint_state_time = float("nan")
        self.latest_joint_state_wall = 0.0
        self.latest_action = np.full(16, np.nan, dtype=np.float64)
        self.latest_action_valid = np.zeros(16, dtype=np.bool_)
        self.latest_gripper = np.full(2, np.nan, dtype=np.float64)

        self.samples: List[Dict[str, np.ndarray]] = []
        self.camera_counts: Dict[str, int] = {name: 0 for name in self.camera_names}
        self.camera_timestamps: Dict[str, List[float]] = {name: [] for name in self.camera_names}
        self.camera_ros_timestamps: Dict[str, List[float]] = {name: [] for name in self.camera_names}
        self.topic_wall_times: Dict[str, float] = {}
        self.topic_message_counts: Dict[str, int] = {}
        self.last_rate_check_wall = time.monotonic()
        self.last_rate_check_counts: Dict[str, int] = {}

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        default_qos = QoSProfile(depth=10)

        self.joint_sub = self.create_subscription(
            JointState, self.joint_state_topic, self._joint_state_cb, sensor_qos
        )
        self.left_cmd_sub = self.create_subscription(
            Float64MultiArray, self.left_joint_command_topic, self._left_cmd_cb, default_qos
        )
        self.right_cmd_sub = self.create_subscription(
            Float64MultiArray, self.right_joint_command_topic, self._right_cmd_cb, default_qos
        )
        self.neck_cmd_sub = self.create_subscription(
            Float64MultiArray, self.neck_joint_command_topic, self._neck_cmd_cb, default_qos
        )
        self.left_gripper_sub = self.create_subscription(
            Bool, self.left_gripper_topic, self._left_gripper_cb, default_qos
        )
        self.right_gripper_sub = self.create_subscription(
            Bool, self.right_gripper_topic, self._right_gripper_cb, default_qos
        )

        self.camera_subs = []
        for name, topic in zip(self.camera_names, self.camera_topics):
            self.camera_subs.append(
                self.create_subscription(
                    Image,
                    topic,
                    lambda msg, camera_name=name, camera_topic=topic: self._image_cb(
                        camera_name, camera_topic, msg
                    ),
                    sensor_qos,
                )
            )

        self.start_srv = self.create_service(Trigger, "~/start", self._start_cb)
        self.stop_srv = self.create_service(SetBool, "~/stop", self._stop_cb)
        self.timer = self.create_timer(1.0 / self.sample_rate_hz, self._sample_once)
        self.warn_timer = self.create_timer(1.0, self._warn_if_stale)

        self.get_logger().info("Demo recorder ready.")
        self.get_logger().info(f"Save root: {self.save_root}")
        self.get_logger().info(f"Cameras: {dict(zip(self.camera_names, self.camera_topics))}")

    def _declare_parameters(self) -> None:
        self.declare_parameter("save_root", "data/imitation_raw")
        self.declare_parameter("sample_rate_hz", 10.0)
        self.declare_parameter("min_joint_state_hz", 5.0)
        self.declare_parameter("min_camera_hz", 5.0)
        self.declare_parameter("stale_topic_warn_sec", 1.0)
        self.declare_parameter("image_format", "jpg")
        self.declare_parameter("jpeg_quality", 92)
        self.declare_parameter("action_mode", "joint_position_targets")
        self.declare_parameter("require_joint_state_before_start", True)
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("left_joint_command_topic", "/left_joint_command")
        self.declare_parameter("right_joint_command_topic", "/right_joint_command")
        self.declare_parameter("neck_joint_command_topic", "/neck_joint_command")
        self.declare_parameter("left_gripper_topic", "/open_left_gripper")
        self.declare_parameter("right_gripper_topic", "/open_right_gripper")
        self.declare_parameter(
            "left_joint_names",
            [
                "left_base_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_pitch_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
            ],
        )
        self.declare_parameter(
            "right_joint_names",
            [
                "right_base_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_pitch_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
        )
        self.declare_parameter("neck_joint_names", ["neck_pitch_joint", "neck_yaw_joint"])
        self.declare_parameter("gripper_joint_names", ["left_gripper1_joint", "right_gripper1_joint"])
        self.declare_parameter("camera_names", ["rgb_head"])
        self.declare_parameter("camera_topics", ["/head_camera/image"])
        self.declare_parameter("camera_enabled", [True])
        self.declare_parameter("optional_camera_names", ["rgb_wrist_left", "rgb_wrist_right"])
        self.declare_parameter("optional_camera_topics", ["/left_camera/image", "/right_camera/image"])
        self.declare_parameter("optional_camera_enabled", [False, False])

    def _load_parameters(self) -> None:
        self.save_root = Path(self.get_parameter("save_root").value)
        if not self.save_root.is_absolute():
            self.save_root = Path.cwd() / self.save_root
        self.sample_rate_hz = float(self.get_parameter("sample_rate_hz").value)
        self.min_joint_state_hz = float(self.get_parameter("min_joint_state_hz").value)
        self.min_camera_hz = float(self.get_parameter("min_camera_hz").value)
        self.stale_topic_warn_sec = float(self.get_parameter("stale_topic_warn_sec").value)
        self.image_format = str(self.get_parameter("image_format").value).lower()
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.action_mode = str(self.get_parameter("action_mode").value)
        self.require_joint_state_before_start = bool(
            self.get_parameter("require_joint_state_before_start").value
        )
        self.joint_state_topic = str(self.get_parameter("joint_state_topic").value)
        self.left_joint_command_topic = str(self.get_parameter("left_joint_command_topic").value)
        self.right_joint_command_topic = str(self.get_parameter("right_joint_command_topic").value)
        self.neck_joint_command_topic = str(self.get_parameter("neck_joint_command_topic").value)
        self.left_gripper_topic = str(self.get_parameter("left_gripper_topic").value)
        self.right_gripper_topic = str(self.get_parameter("right_gripper_topic").value)
        self.left_joint_names = list(self.get_parameter("left_joint_names").value)
        self.right_joint_names = list(self.get_parameter("right_joint_names").value)
        self.neck_joint_names = list(self.get_parameter("neck_joint_names").value)
        self.gripper_joint_names = list(self.get_parameter("gripper_joint_names").value)

        camera_names = list(self.get_parameter("camera_names").value)
        camera_topics = list(self.get_parameter("camera_topics").value)
        camera_enabled = list(self.get_parameter("camera_enabled").value)
        opt_names = list(self.get_parameter("optional_camera_names").value)
        opt_topics = list(self.get_parameter("optional_camera_topics").value)
        opt_enabled = list(self.get_parameter("optional_camera_enabled").value)
        camera_names += opt_names
        camera_topics += opt_topics
        camera_enabled += opt_enabled
        self.camera_names = []
        self.camera_topics = []
        for name, topic, enabled in zip(camera_names, camera_topics, camera_enabled):
            if enabled:
                self.camera_names.append(str(name))
                self.camera_topics.append(str(topic))

    def _joint_state_cb(self, msg: JointState) -> None:
        self.latest_joint_state = msg
        self.latest_joint_state_time = stamp_to_float(msg.header.stamp)
        self.latest_joint_state_wall = time.monotonic()
        self.topic_wall_times[self.joint_state_topic] = self.latest_joint_state_wall
        self.topic_message_counts[self.joint_state_topic] = self.topic_message_counts.get(self.joint_state_topic, 0) + 1

    def _left_cmd_cb(self, msg: Float64MultiArray) -> None:
        self._set_action_slice("left_joint_target", msg.data, self.left_joint_command_topic)

    def _right_cmd_cb(self, msg: Float64MultiArray) -> None:
        self._set_action_slice("right_joint_target", msg.data, self.right_joint_command_topic)

    def _neck_cmd_cb(self, msg: Float64MultiArray) -> None:
        self._set_action_slice("neck_joint_target", msg.data, self.neck_joint_command_topic)

    def _left_gripper_cb(self, msg: Bool) -> None:
        self.latest_action[14] = 1.0 if msg.data else 0.0
        self.latest_action_valid[14] = True
        self.latest_gripper[0] = self.latest_action[14]
        self.topic_wall_times[self.left_gripper_topic] = time.monotonic()

    def _right_gripper_cb(self, msg: Bool) -> None:
        self.latest_action[15] = 1.0 if msg.data else 0.0
        self.latest_action_valid[15] = True
        self.latest_gripper[1] = self.latest_action[15]
        self.topic_wall_times[self.right_gripper_topic] = time.monotonic()

    def _set_action_slice(self, component: str, values: List[float], topic: str) -> None:
        start, end = ACTION_COMPONENTS[component]
        expected = end - start
        if len(values) != expected:
            self.get_logger().warn(
                f"Ignoring {topic}: expected {expected} values, received {len(values)}"
            )
            return
        self.latest_action[start:end] = np.asarray(values, dtype=np.float64)
        self.latest_action_valid[start:end] = True
        self.topic_wall_times[topic] = time.monotonic()

    def _image_cb(self, camera_name: str, topic: str, msg: Image) -> None:
        self.topic_wall_times[topic] = time.monotonic()
        self.topic_message_counts[topic] = self.topic_message_counts.get(topic, 0) + 1
        if not self.recording or self.episode_dir is None:
            return
        if self.bridge is None or cv2 is None:
            self.get_logger().warn("cv_bridge and OpenCV are required to save camera JPG frames.")
            return
        camera_dir = self.episode_dir / camera_name
        camera_dir.mkdir(parents=True, exist_ok=True)
        idx = self.camera_counts[camera_name] + 1
        filename = camera_dir / f"frame_{idx:06d}.{self.image_format}"
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            if self.image_format in ("jpg", "jpeg"):
                ok = cv2.imwrite(str(filename), image, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            else:
                ok = cv2.imwrite(str(filename), image)
            if not ok:
                raise RuntimeError(f"cv2.imwrite returned false for {filename}")
        except Exception as exc:
            self.get_logger().warn(f"Failed to save image from {topic}: {exc}")
            return
        self.camera_counts[camera_name] = idx
        self.camera_timestamps[camera_name].append(now_to_float(self.get_clock()))
        self.camera_ros_timestamps[camera_name].append(stamp_to_float(msg.header.stamp))

    def _start_cb(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if self.recording:
            response.success = False
            response.message = f"Already recording {self.episode_dir}"
            return response
        if self.require_joint_state_before_start and self.latest_joint_state is None:
            response.success = False
            response.message = "No /joint_states received yet; refusing to start."
            return response
        self.episode_dir = next_episode_dir(self.save_root)
        for name in self.camera_names:
            (self.episode_dir / name).mkdir(parents=True, exist_ok=True)
        self.samples = []
        self.camera_counts = {name: 0 for name in self.camera_names}
        self.camera_timestamps = {name: [] for name in self.camera_names}
        self.camera_ros_timestamps = {name: [] for name in self.camera_names}
        self.episode_start_wall = time.time()
        self.episode_start_ros = now_to_float(self.get_clock())
        self._initialize_action_from_state()
        self.recording = True
        response.success = True
        response.message = str(self.episode_dir)
        self.get_logger().info(f"Started recording: {self.episode_dir}")
        return response

    def _stop_cb(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        if not self.recording:
            response.success = False
            response.message = "Recorder is not running."
            return response
        episode_dir = self.episode_dir
        success = bool(request.data)
        self.recording = False
        self._flush_episode(success)
        response.success = True
        response.message = str(episode_dir)
        self.get_logger().info(f"Stopped recording: {episode_dir}, success={success}")
        return response

    def _initialize_action_from_state(self) -> None:
        if self.latest_joint_state is None:
            return
        name_to_pos = dict(zip(self.latest_joint_state.name, self.latest_joint_state.position))
        self._initialize_slice_from_names("left_joint_target", self.left_joint_names, name_to_pos)
        self._initialize_slice_from_names("right_joint_target", self.right_joint_names, name_to_pos)
        self._initialize_slice_from_names("neck_joint_target", self.neck_joint_names, name_to_pos)
        for idx, name in enumerate(self.gripper_joint_names[:2]):
            if name in name_to_pos:
                self.latest_gripper[idx] = float(name_to_pos[name])

    def _initialize_slice_from_names(self, component: str, names: List[str], name_to_pos: Dict[str, float]) -> None:
        start, end = ACTION_COMPONENTS[component]
        if len(names) != (end - start):
            return
        values = []
        for name in names:
            if name not in name_to_pos:
                return
            values.append(name_to_pos[name])
        self.latest_action[start:end] = np.asarray(values, dtype=np.float64)
        self.latest_action_valid[start:end] = True

    def _sample_once(self) -> None:
        if not self.recording or self.latest_joint_state is None:
            return
        joint_pos, joint_vel = self._ordered_joint_arrays(self.latest_joint_state)
        sample = {
            "timestamp": np.asarray(now_to_float(self.get_clock()), dtype=np.float64),
            "joint_state_timestamp": np.asarray(self.latest_joint_state_time, dtype=np.float64),
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "action": self.latest_action.copy(),
            "action_valid": self.latest_action_valid.copy(),
            "gripper": self.latest_gripper.copy(),
        }
        self.samples.append(sample)

    def _ordered_joint_arrays(self, msg: JointState) -> tuple[np.ndarray, np.ndarray]:
        names = self.left_joint_names + self.right_joint_names + self.neck_joint_names + self.gripper_joint_names[:2]
        name_to_pos = dict(zip(msg.name, msg.position))
        name_to_vel = dict(zip(msg.name, msg.velocity))
        pos = np.asarray([name_to_pos.get(name, np.nan) for name in names], dtype=np.float64)
        vel = np.asarray([name_to_vel.get(name, np.nan) for name in names], dtype=np.float64)
        return pos, vel

    def _flush_episode(self, success: bool) -> None:
        if self.episode_dir is None:
            return
        episode_dir = self.episode_dir
        n = len(self.samples)
        if n == 0:
            timestamps = np.zeros((0,), dtype=np.float64)
            joint_state_timestamps = np.zeros((0,), dtype=np.float64)
            joint_pos = np.zeros((0, 16), dtype=np.float64)
            joint_vel = np.zeros((0, 16), dtype=np.float64)
            actions = np.zeros((0, 16), dtype=np.float64)
            action_valid = np.zeros((0, 16), dtype=np.bool_)
            gripper = np.zeros((0, 2), dtype=np.float64)
        else:
            timestamps = np.asarray([s["timestamp"] for s in self.samples], dtype=np.float64)
            joint_state_timestamps = np.asarray(
                [s["joint_state_timestamp"] for s in self.samples], dtype=np.float64
            )
            joint_pos = np.stack([s["joint_pos"] for s in self.samples])
            joint_vel = np.stack([s["joint_vel"] for s in self.samples])
            actions = np.stack([s["action"] for s in self.samples])
            action_valid = np.stack([s["action_valid"] for s in self.samples])
            gripper = np.stack([s["gripper"] for s in self.samples])
        np.save(episode_dir / "timestamps.npy", timestamps)
        np.save(episode_dir / "joint_state_timestamps.npy", joint_state_timestamps)
        np.save(episode_dir / "joint_pos.npy", joint_pos)
        np.save(episode_dir / "joint_vel.npy", joint_vel)
        np.save(episode_dir / "actions.npy", actions)
        np.save(episode_dir / "action_valid.npy", action_valid)
        np.save(episode_dir / "gripper.npy", gripper)
        for name in self.camera_names:
            np.save(episode_dir / f"{name}_timestamps.npy", np.asarray(self.camera_timestamps[name], dtype=np.float64))
            np.save(
                episode_dir / f"{name}_ros_timestamps.npy",
                np.asarray(self.camera_ros_timestamps[name], dtype=np.float64),
            )
        duration = 0.0
        if n >= 2:
            duration = float(timestamps[-1] - timestamps[0])
        meta = {
            "schema_version": "0.1.0",
            "created_wall_time": self.episode_start_wall,
            "start_ros_time": self.episode_start_ros,
            "end_ros_time": now_to_float(self.get_clock()),
            "duration_sec": duration,
            "sample_rate_hz": self.sample_rate_hz,
            "num_samples": n,
            "action_mode": self.action_mode,
            "action_components": ACTION_COMPONENTS,
            "joint_state_topic": self.joint_state_topic,
            "command_topics": {
                "left_joint": self.left_joint_command_topic,
                "right_joint": self.right_joint_command_topic,
                "neck_joint": self.neck_joint_command_topic,
                "left_gripper": self.left_gripper_topic,
                "right_gripper": self.right_gripper_topic,
            },
            "camera_topics": dict(zip(self.camera_names, self.camera_topics)),
            "camera_frame_counts": self.camera_counts,
            "joint_names": self.left_joint_names + self.right_joint_names + self.neck_joint_names + self.gripper_joint_names[:2],
            "left_joint_names": self.left_joint_names,
            "right_joint_names": self.right_joint_names,
            "neck_joint_names": self.neck_joint_names,
            "gripper_joint_names": self.gripper_joint_names[:2],
        }
        write_json(episode_dir / "meta.json", meta)
        write_json(episode_dir / "success.json", {"success": success})

    def _warn_if_stale(self) -> None:
        now = time.monotonic()
        topics = [self.joint_state_topic] + self.camera_topics
        for topic in topics:
            last = self.topic_wall_times.get(topic)
            if last is None:
                self.get_logger().warn(f"No messages received yet on {topic}")
                continue
            age = now - last
            if age > self.stale_topic_warn_sec:
                self.get_logger().warn(f"Topic {topic} is stale: {age:.2f}s since last message")
        elapsed = now - self.last_rate_check_wall
        if elapsed <= 0.0:
            return
        min_rates = {self.joint_state_topic: self.min_joint_state_hz}
        for topic in self.camera_topics:
            min_rates[topic] = self.min_camera_hz
        for topic, min_rate in min_rates.items():
            count = self.topic_message_counts.get(topic, 0)
            last_count = self.last_rate_check_counts.get(topic, 0)
            rate = float(count - last_count) / elapsed
            if count > 0 and rate < min_rate:
                self.get_logger().warn(
                    f"Topic {topic} rate is low: {rate:.2f} Hz < expected {min_rate:.2f} Hz"
                )
            self.last_rate_check_counts[topic] = count
        self.last_rate_check_wall = now


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DemoRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.recording:
            node.get_logger().warn("Interrupted while recording; saving as failure.")
            node.recording = False
            node._flush_episode(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
