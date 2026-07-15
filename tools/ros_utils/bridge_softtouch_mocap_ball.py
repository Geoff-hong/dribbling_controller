#!/usr/bin/env python3
"""Bridge a mocap ball PoseStamped stream to the SoftTouch ball pose/twist topics."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


@dataclass
class TimedPosition:
    stamp_s: float
    x: float
    y: float
    z: float


def stamp_to_seconds(msg: PoseStamped, node: Node) -> float:
    stamp = msg.header.stamp
    if stamp.sec == 0 and stamp.nanosec == 0:
        return node.get_clock().now().nanoseconds * 1.0e-9
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


class SoftTouchMocapBallBridge(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("softtouch_mocap_ball_bridge")
        self.args = args
        self.prev: TimedPosition | None = None
        self.filtered_velocity = [0.0, 0.0, 0.0]

        self.pose_pub = self.create_publisher(PoseStamped, args.output_pose, qos_profile_sensor_data)
        self.twist_pub = self.create_publisher(TwistStamped, args.output_twist, qos_profile_sensor_data)
        self.sub = self.create_subscription(PoseStamped, args.input_pose, self.handle_pose, qos_profile_sensor_data)

        self.get_logger().info(
            f"Bridging mocap ball pose {args.input_pose} -> {args.output_pose}, {args.output_twist}"
        )

    def handle_pose(self, msg: PoseStamped) -> None:
        stamp_s = stamp_to_seconds(msg, self)
        current = TimedPosition(
            stamp_s=stamp_s,
            x=float(msg.pose.position.x) * self.args.position_scale + self.args.position_offset[0],
            y=float(msg.pose.position.y) * self.args.position_scale + self.args.position_offset[1],
            z=float(msg.pose.position.z) * self.args.position_scale + self.args.position_offset[2],
        )

        pose = PoseStamped()
        pose.header = msg.header
        if self.args.frame_id:
            pose.header.frame_id = self.args.frame_id
        pose.pose = msg.pose
        pose.pose.position.x = current.x
        pose.pose.position.y = current.y
        pose.pose.position.z = current.z

        velocity = [0.0, 0.0, 0.0]
        if self.prev is not None:
            dt = current.stamp_s - self.prev.stamp_s
            if self.args.min_dt <= dt <= self.args.max_dt:
                raw_velocity = [
                    (current.x - self.prev.x) / dt,
                    (current.y - self.prev.y) / dt,
                    (current.z - self.prev.z) / dt,
                ]
                if all(math.isfinite(value) for value in raw_velocity):
                    alpha = self.args.velocity_alpha
                    self.filtered_velocity = [
                        alpha * raw + (1.0 - alpha) * old
                        for raw, old in zip(raw_velocity, self.filtered_velocity)
                    ]
                    velocity = list(self.filtered_velocity)
            else:
                self.filtered_velocity = [0.0, 0.0, 0.0]
        self.prev = current

        twist = TwistStamped()
        twist.header = pose.header
        twist.twist.linear.x = velocity[0]
        twist.twist.linear.y = velocity[1]
        twist.twist.linear.z = velocity[2]

        self.pose_pub.publish(pose)
        self.twist_pub.publish(twist)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pose", default="/mocap/ball/pose", help="Input mocap PoseStamped topic.")
    parser.add_argument("--output-pose", default="/softtouch/ball/pose", help="Output SoftTouch PoseStamped topic.")
    parser.add_argument("--output-twist", default="/softtouch/ball/twist", help="Output SoftTouch TwistStamped topic.")
    parser.add_argument("--frame-id", default="world", help="Override output frame_id. Empty string preserves input.")
    parser.add_argument("--position-scale", type=float, default=1.0, help="Scale mocap position before publishing.")
    parser.add_argument(
        "--position-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Add xyz offset after scaling mocap position.",
    )
    parser.add_argument("--velocity-alpha", type=float, default=0.35, help="EMA alpha for differentiated velocity.")
    parser.add_argument("--min-dt", type=float, default=1.0e-4, help="Minimum valid pose dt for velocity estimation.")
    parser.add_argument("--max-dt", type=float, default=0.2, help="Maximum valid pose dt before velocity resets to zero.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.velocity_alpha = max(0.0, min(1.0, float(args.velocity_alpha)))
    args.position_offset = tuple(float(value) for value in args.position_offset)

    rclpy.init()
    node = SoftTouchMocapBallBridge(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
