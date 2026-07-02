#!/usr/bin/env python3
"""Headless fall/robustness monitor for the SoftTouch dribble ROS 2 sim2sim.

Subscribes to the base pose the MuJoCo bridge publishes and, for a fixed wall-clock
window, tracks the base height. It then writes a one-line JSON verdict (fell yes/no,
min height, survival time) so the DR sweep can aggregate pass/fail across variants
without a GUI.

    ros2 run ... (or) python tools/fall_monitor.py --seconds 35 --out result.json --label dr_003

The robot stands at ~0.75 m; a fall is min base height dropping below --fall-z after
an initial --settle-s window (which skips the reset/hold transient).
"""
import argparse
import json
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped


class FallMonitor(Node):
    def __init__(self, topic: str, fall_z: float):
        super().__init__("softtouch_fall_monitor")
        self.fall_z = fall_z
        self.t0 = None            # wall time of first message
        self.min_z = float("inf")
        self.last_z = float("nan")
        self.fall_t = None        # seconds-since-t0 of first sub-threshold sample
        self.n = 0
        # The bridge publishes base state BEST_EFFORT; a BEST_EFFORT sub is compatible
        # with both best-effort and reliable publishers.
        qos = QoSProfile(depth=20, reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(PoseStamped, topic, self._cb, qos)

    def _cb(self, msg: PoseStamped):
        now = time.monotonic()
        if self.t0 is None:
            self.t0 = now
        z = msg.pose.position.z
        self.last_z = z
        self.n += 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=15.0,
                    help="dribble window recorded, counted from the FIRST base-pose message")
    ap.add_argument("--max-wait", type=float, default=25.0,
                    help="give up if no base-pose message arrives within this many seconds")
    ap.add_argument("--settle-s", type=float, default=4.0, help="ignore falls before this (reset transient)")
    ap.add_argument("--fall-z", type=float, default=0.45, help="base height below this = fallen")
    ap.add_argument("--topic", default="/softtouch/base/pose")
    ap.add_argument("--out", default="", help="write JSON verdict here (else stdout only)")
    ap.add_argument("--label", default="", help="tag copied into the JSON (e.g. variant id)")
    args = ap.parse_args()

    rclpy.init()
    node = FallMonitor(args.topic, args.fall_z)
    start = time.monotonic()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if node.t0 is None:
                if now - start > args.max_wait:
                    break  # sim never published base pose
                continue
            elapsed = now - node.t0  # window counts from the first message
            if elapsed >= args.seconds:
                break
            z = node.last_z
            if z == z:  # not NaN
                if elapsed >= args.settle_s:
                    node.min_z = min(node.min_z, z)
                    if node.fall_t is None and z < args.fall_z:
                        node.fall_t = elapsed
    finally:
        observed = 0.0 if node.t0 is None else (time.monotonic() - node.t0)
        fell = node.fall_t is not None
        result = dict(
            label=args.label,
            got_data=node.n > 0,
            samples=node.n,
            min_z=None if node.min_z == float("inf") else round(node.min_z, 4),
            last_z=None if node.last_z != node.last_z else round(node.last_z, 4),
            fell=fell,
            survival_s=round(node.fall_t, 2) if fell else round(observed, 2),
            observed_s=round(observed, 2),
        )
        node.destroy_node()
        rclpy.shutdown()
        line = json.dumps(result)
        print("[fall_monitor] " + line)
        if args.out:
            with open(args.out, "w") as f:
                f.write(line + "\n")


if __name__ == "__main__":
    main()
