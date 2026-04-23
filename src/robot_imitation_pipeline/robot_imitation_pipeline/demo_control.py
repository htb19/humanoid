#!/usr/bin/env python3
import argparse
import sys

import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool, Trigger


class DemoControlClient(Node):
    def __init__(self) -> None:
        super().__init__("demo_control")
        self.start_client = self.create_client(Trigger, "/demo_recorder/start")
        self.stop_client = self.create_client(SetBool, "/demo_recorder/stop")

    def start(self) -> int:
        if not self.start_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service /demo_recorder/start is not available.")
            return 2
        future = self.start_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            self.get_logger().error("Start service returned no result.")
            return 2
        print(result.message)
        return 0 if result.success else 1

    def stop(self, success: bool) -> int:
        if not self.stop_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("Service /demo_recorder/stop is not available.")
            return 2
        request = SetBool.Request()
        request.data = success
        future = self.stop_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        if result is None:
            self.get_logger().error("Stop service returned no result.")
            return 2
        print(result.message)
        return 0 if result.success else 1


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Control the demo recorder services.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("start")
    stop = sub.add_parser("stop")
    group = stop.add_mutually_exclusive_group(required=True)
    group.add_argument("--success", action="store_true", help="Save the episode as successful.")
    group.add_argument("--failure", action="store_true", help="Save the episode as failed.")
    args = parser.parse_args(argv)

    rclpy.init()
    node = DemoControlClient()
    try:
        if args.command == "start":
            code = node.start()
        else:
            code = node.stop(success=bool(args.success))
    finally:
        node.destroy_node()
        rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
