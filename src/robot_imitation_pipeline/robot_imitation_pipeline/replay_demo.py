#!/usr/bin/env python3
import argparse
from pathlib import Path
import time

import numpy as np

from robot_imitation_pipeline.io_utils import load_episode_arrays, maybe_load_yaml, nested_get, read_json


def _load_replay_config(path: Path | None) -> dict:
    data = maybe_load_yaml(path)
    params = nested_get(data, ["replay", "ros__parameters"], data)
    return params or {}


def dry_run(episode_dir: Path, print_limit: int) -> None:
    arrays = load_episode_arrays(episode_dir)
    actions = arrays["actions"]
    timestamps = arrays["timestamps"]
    meta = read_json(episode_dir / "meta.json")
    print(f"Dry-run replay: {episode_dir}")
    print(f"Action mode: {meta.get('action_mode')}")
    print(f"Samples: {len(actions)}")
    for idx, action in enumerate(actions[:print_limit]):
        t = timestamps[idx] - timestamps[0] if len(timestamps) else 0.0
        print(
            f"[{idx:05d}] t={t:.3f} "
            f"left={np.array2string(action[0:6], precision=3)} "
            f"right={np.array2string(action[6:12], precision=3)} "
            f"neck={np.array2string(action[12:14], precision=3)} "
            f"gripper={np.array2string(action[14:16], precision=1)}"
        )
    if len(actions) > print_limit:
        print(f"... {len(actions) - print_limit} more samples omitted")


def execute_on_robot(episode_dir: Path, config: dict, rate_hz: float) -> None:
    import rclpy
    from example_interfaces.msg import Bool, Float64MultiArray
    from rclpy.node import Node

    class ReplayNode(Node):
        def __init__(self) -> None:
            super().__init__("demo_replay")
            self.left_pub = self.create_publisher(
                Float64MultiArray, config.get("left_joint_command_topic", "/left_joint_command"), 10
            )
            self.right_pub = self.create_publisher(
                Float64MultiArray, config.get("right_joint_command_topic", "/right_joint_command"), 10
            )
            self.neck_pub = self.create_publisher(
                Float64MultiArray, config.get("neck_joint_command_topic", "/neck_joint_command"), 10
            )
            self.left_gripper_pub = self.create_publisher(
                Bool, config.get("left_gripper_topic", "/open_left_gripper"), 10
            )
            self.right_gripper_pub = self.create_publisher(
                Bool, config.get("right_gripper_topic", "/open_right_gripper"), 10
            )

    arrays = load_episode_arrays(episode_dir)
    actions = arrays["actions"]
    valid = arrays.get("action_valid", np.ones_like(actions, dtype=bool))
    rclpy.init()
    node = ReplayNode()
    period = 1.0 / rate_hz
    try:
        for idx, action in enumerate(actions):
            if valid.ndim == 2 and not valid[idx].all():
                node.get_logger().warn(f"Skipping invalid action sample {idx}")
                continue
            left = Float64MultiArray()
            left.data = action[0:6].astype(float).tolist()
            right = Float64MultiArray()
            right.data = action[6:12].astype(float).tolist()
            neck = Float64MultiArray()
            neck.data = action[12:14].astype(float).tolist()
            left_gripper = Bool()
            left_gripper.data = bool(action[14] >= 0.5)
            right_gripper = Bool()
            right_gripper.data = bool(action[15] >= 0.5)
            node.left_pub.publish(left)
            node.right_pub.publish(right)
            node.neck_pub.publish(neck)
            node.left_gripper_pub.publish(left_gripper)
            node.right_gripper_pub.publish(right_gripper)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Replay or dry-run recorded demo actions.")
    parser.add_argument("episode", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--execute-on-robot", action="store_true")
    parser.add_argument("--rate-hz", type=float, default=None)
    parser.add_argument("--dry-run-print-limit", type=int, default=None)
    args = parser.parse_args(argv)

    config = _load_replay_config(args.config)
    configured_execute = bool(config.get("execute_on_robot", False))
    execute = bool(args.execute_on_robot and configured_execute)
    if args.execute_on_robot and not configured_execute:
        raise SystemExit(
            "Refusing to execute because replay config has execute_on_robot=false. "
            "Set it true in an explicit config file and pass --execute-on-robot."
        )
    if execute:
        execute_on_robot(args.episode, config, args.rate_hz or float(config.get("rate_hz", 10.0)))
    else:
        dry_run(args.episode, args.dry_run_print_limit or int(config.get("dry_run_print_limit", 20)))


if __name__ == "__main__":
    main()
