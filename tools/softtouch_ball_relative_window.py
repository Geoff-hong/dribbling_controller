#!/usr/bin/python3
import argparse
import math
import threading
import time
import tkinter as tk

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node


def q_normalize(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n <= 1.0e-12 or not math.isfinite(n):
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def q_conj(q):
    x, y, z, w = q
    return (-x, -y, -z, w)


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def rotate(q, v):
    out = q_mul(q_mul(q, (v[0], v[1], v[2], 0.0)), q_conj(q))
    return (out[0], out[1], out[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


class RelativeBallNode(Node):
    def __init__(self, args):
        super().__init__("softtouch_ball_relative_window")
        self.args = args
        self.lock = threading.Lock()
        self.frame = None
        self.ball = None
        qos = 1
        self.create_subscription(PoseStamped, args.frame_topic, self.on_frame, qos)
        self.create_subscription(PoseStamped, args.ball_topic, self.on_ball, qos)
        self.get_logger().info(
            "SoftTouch ball relative window: "
            f"obs_frame={args.frame_topic}, ball={args.ball_topic}, target={args.target}"
        )

    def on_frame(self, msg):
        pose = msg.pose
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.0e-9
        with self.lock:
            self.frame = (
                (pose.position.x, pose.position.y, pose.position.z),
                q_normalize((pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)),
                stamp,
            )

    def on_ball(self, msg):
        pose = msg.pose
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1.0e-9
        with self.lock:
            self.ball = ((pose.position.x, pose.position.y, pose.position.z), stamp)

    def snapshot(self):
        with self.lock:
            frame = self.frame
            ball = self.ball
        if frame is None or ball is None:
            return None

        frame_pos_w, frame_q_w, frame_stamp = frame
        ball_pos_w, ball_stamp = ball

        ball_rel_frame = rotate(q_conj(frame_q_w), sub(ball_pos_w, frame_pos_w))
        now = time.time()
        age = max(now - frame_stamp, now - ball_stamp)
        target = tuple(self.args.target)
        err = sub(ball_rel_frame, target)
        return {
            "ball_rel": ball_rel_frame,
            "err": err,
            "target": target,
            "frame_pos_w": frame_pos_w,
            "ball_pos_w": ball_pos_w,
            "age": age,
        }


class RelativeBallWindow:
    def __init__(self, node):
        self.node = node
        self.root = tk.Tk()
        self.root.title("SoftTouch Ball Relative")
        self.root.geometry("520x300")
        self.root.configure(bg="#111111")
        self.font_big = ("DejaVu Sans Mono", 24, "bold")
        self.font_mid = ("DejaVu Sans Mono", 16)
        self.font_small = ("DejaVu Sans Mono", 12)

        tx, ty, tz = self.node.args.target
        self.title = tk.Label(
            self.root,
            text=f"ball_pos_b  target=({tx:+.3f}, {ty:+.3f}, {tz:+.3f})",
            fg="#dddddd",
            bg="#111111",
            font=self.font_small,
        )
        self.title.pack(pady=(10, 4))
        self.value = tk.Label(self.root, text="waiting for mocap...", fg="#ffffff", bg="#111111", font=self.font_big)
        self.value.pack(pady=4)
        self.error = tk.Label(self.root, text="", fg="#ffcc66", bg="#111111", font=self.font_mid)
        self.error.pack(pady=4)
        self.hint = tk.Label(self.root, text="", fg="#cccccc", bg="#111111", font=self.font_mid)
        self.hint.pack(pady=4)
        self.detail = tk.Label(self.root, text="", fg="#999999", bg="#111111", font=self.font_small)
        self.detail.pack(pady=4)
        self.update()

    def update(self):
        snap = self.node.snapshot()
        if snap is None:
            self.value.config(text="waiting for mocap...", fg="#ffffff")
            self.error.config(text="")
            self.hint.config(text="")
            self.detail.config(text="")
        else:
            x, y, z = snap["ball_rel"]
            ex, ey, ez = snap["err"]
            xy_error = math.sqrt(ex * ex + ey * ey)
            ok = abs(ex) < 0.03 and abs(ey) < 0.03
            warn = xy_error < 0.07
            color = "#61d36b" if ok else ("#ffcc66" if warn else "#ff6666")
            self.value.config(text=f"x={x:+.3f}  y={y:+.3f}  z={z:+.3f}", fg=color)
            self.error.config(text=f"err: dx={ex:+.3f}  dy={ey:+.3f}  dz={ez:+.3f}", fg=color)

            hints = []
            if abs(ex) >= 0.03:
                hints.append("move ball forward" if ex < 0 else "move ball backward")
            if abs(ey) >= 0.03:
                hints.append("move ball right" if ey > 0 else "move ball left")
            self.hint.config(text=" / ".join(hints) if hints else "XY aligned")

            px, py, pz = snap["frame_pos_w"]
            bx, by, bz = snap["ball_pos_w"]
            self.detail.config(
                text=f"+x forward, +y left | age={snap['age']*1000:.0f} ms\n"
                     f"frame_w=({px:+.2f},{py:+.2f},{pz:+.2f})  ball_w=({bx:+.2f},{by:+.2f},{bz:+.2f})"
            )
        self.root.after(100, self.update)

    def run(self):
        self.root.mainloop()


def main():
    parser = argparse.ArgumentParser(description="Show SoftTouch ball position in the policy observation frame.")
    parser.add_argument("--frame-topic", default="/softtouch/mocap/chest/pose")
    parser.add_argument("--ball-topic", default="/softtouch/mocap/ball/pose")
    parser.add_argument(
        "--target",
        type=float,
        nargs=3,
        default=(0.577, 0.0, -0.852),
        metavar=("X", "Y", "Z"),
        help="Desired ball_pos_b in the policy obs-frame. Default approximates the current MuJoCo standby reset.",
    )
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = RelativeBallNode(args)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    try:
        RelativeBallWindow(node).run()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
