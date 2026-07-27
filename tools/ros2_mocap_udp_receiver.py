#!/usr/bin/python3
import argparse
import math
import select
import socket
import struct
import threading
import time

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node


PACKET = struct.Struct("<4sBiqddddddddd")
MAGIC = b"STMC"
VERSION = 1


def time_msg_from_sec(seconds):
    if seconds <= 0.0 or not math.isfinite(seconds):
        seconds = time.time()
    sec = int(math.floor(seconds))
    nanosec = int(round((seconds - sec) * 1.0e9))
    if nanosec >= 1000000000:
        sec += 1
        nanosec -= 1000000000
    msg = Time()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


def normalize_quaternion(q):
    w, x, y, z = q
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12 or not math.isfinite(norm):
        return 1.0, 0.0, 0.0, 0.0
    return w / norm, x / norm, y / norm, z / norm


def quat_conj(q):
    w, x, y, z = q
    return w, -x, -y, -z


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def rotate_body_to_world(q, v):
    w, x, y, z = q
    vx, vy, vz = v
    out = quat_mul(quat_mul((w, x, y, z), (0.0, vx, vy, vz)), quat_conj((w, x, y, z)))
    return out[1], out[2], out[3]


def angular_velocity_world(prev_q, q, dt):
    if dt <= 1.0e-6:
        return 0.0, 0.0, 0.0
    dq = normalize_quaternion(quat_mul(q, quat_conj(prev_q)))
    w, x, y, z = dq
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    angle = 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), max(-1.0, min(1.0, w)))
    if angle < 1.0e-9:
        return 0.0, 0.0, 0.0
    s = math.sin(0.5 * angle)
    if abs(s) < 1.0e-9:
        return 0.0, 0.0, 0.0
    axis = (x / s, y / s, z / s)
    return axis[0] * angle / dt, axis[1] * angle / dt, axis[2] * angle / dt


def quat_angle(prev_q, q):
    dq = normalize_quaternion(quat_mul(q, quat_conj(prev_q)))
    w, x, y, z = dq
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return 2.0 * math.atan2(math.sqrt(x * x + y * y + z * z), max(-1.0, min(1.0, w)))


class MocapUdpReceiver(Node):
    def __init__(self, args):
        super().__init__("softtouch_ros2_mocap_udp_receiver")
        self.args = args
        self.pose_pub = self.create_publisher(PoseStamped, args.pose_topic, 1)
        self.twist_pub = self.create_publisher(TwistStamped, args.twist_topic, 1)
        self.prev = None
        self.count = 0
        self.reject_count = 0
        self.last_log_time = 0.0
        self.last_reject_log_time = 0.0
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.thread.start()
        self.get_logger().info(
            f"SoftTouch ROS2 mocap UDP receiver: 0.0.0.0:{args.port} -> "
            f"{args.pose_topic}, {args.twist_topic}, frame_id={args.frame_id}, "
            f"offset_world={tuple(args.position_offset_world)}, offset_body={tuple(args.position_offset_body)}, "
            f"max_position_step_m={args.max_position_step_m}, "
            f"max_orientation_step_deg={args.max_orientation_step_deg}, "
            f"jump_check_max_dt={args.jump_check_max_dt}"
        )

    def destroy_node(self):
        self.stopped.set()
        super().destroy_node()

    def _recv_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)
        sock.bind((self.args.bind, self.args.port))
        sock.setblocking(False)

        while rclpy.ok() and not self.stopped.is_set():
            ready, _, _ = select.select([sock], [], [], 0.05)
            if not ready:
                continue
            latest = None
            while True:
                try:
                    latest = sock.recv(PACKET.size)
                except BlockingIOError:
                    break
            if latest is None or len(latest) != PACKET.size:
                continue
            self._handle_packet(latest)

    def _handle_packet(self, data):
        unpacked = PACKET.unpack(data)
        magic, version, vehicle_id, seq, stamp, x, y, z, qw, qx, qy, qz, send_time = unpacked
        if magic != MAGIC or version != VERSION:
            return
        q = normalize_quaternion((qw, qx, qy, qz))
        body_offset_world = rotate_body_to_world(q, self.args.position_offset_body)
        x += self.args.position_offset_world[0] + body_offset_world[0]
        y += self.args.position_offset_world[1] + body_offset_world[1]
        z += self.args.position_offset_world[2] + body_offset_world[2]

        if self.prev is not None:
            prev_stamp, prev_pos, prev_q = self.prev
            dt = stamp - prev_stamp
            if 1.0e-6 < dt <= self.args.jump_check_max_dt:
                position_step = math.sqrt(
                    (x - prev_pos[0]) ** 2 + (y - prev_pos[1]) ** 2 + (z - prev_pos[2]) ** 2
                )
                orientation_step_deg = math.degrees(quat_angle(prev_q, q))
                reject_reason = None
                if self.args.max_position_step_m > 0.0 and position_step > self.args.max_position_step_m:
                    reject_reason = f"position step {position_step:.3f} m"
                if (
                    self.args.max_orientation_step_deg > 0.0
                    and orientation_step_deg > self.args.max_orientation_step_deg
                ):
                    reject_reason = f"orientation step {orientation_step_deg:.1f} deg"
                if reject_reason is not None:
                    self.reject_count += 1
                    now = time.time()
                    if now - self.last_reject_log_time > 1.0:
                        self.get_logger().warn(
                            f"reject mocap vehicle={vehicle_id} seq={seq}: {reject_reason} over {dt:.4f}s; "
                            f"rejects={self.reject_count}"
                        )
                        self.last_reject_log_time = now
                    return

        pose = PoseStamped()
        pose.header.stamp = time_msg_from_sec(stamp)
        pose.header.frame_id = self.args.frame_id
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = q[0]
        pose.pose.orientation.x = q[1]
        pose.pose.orientation.y = q[2]
        pose.pose.orientation.z = q[3]

        twist = TwistStamped()
        twist.header = pose.header
        if self.prev is not None:
            prev_stamp, prev_pos, prev_q = self.prev
            dt = stamp - prev_stamp
            if dt > 1.0e-6:
                twist.twist.linear.x = (x - prev_pos[0]) / dt
                twist.twist.linear.y = (y - prev_pos[1]) / dt
                twist.twist.linear.z = (z - prev_pos[2]) / dt
                wx, wy, wz = angular_velocity_world(prev_q, q, dt)
                twist.twist.angular.x = wx
                twist.twist.angular.y = wy
                twist.twist.angular.z = wz
        self.prev = (stamp, (x, y, z), q)

        self.pose_pub.publish(pose)
        self.twist_pub.publish(twist)
        self.count += 1
        now = time.time()
        if now - self.last_log_time > 2.0:
            mocap_to_ros1_send_ms = (send_time - stamp) * 1000.0
            udp_to_ros2_receive_ms = (now - send_time) * 1000.0
            mocap_to_ros2_receive_ms = (now - stamp) * 1000.0
            self.get_logger().info(
                f"mocap vehicle={vehicle_id} seq={seq} pos=({x:.3f},{y:.3f},{z:.3f}) "
                f"latency_ms[mocap_to_ros1_send={mocap_to_ros1_send_ms:.2f}, "
                f"udp_to_ros2_receive={udp_to_ros2_receive_ms:.2f}, "
                f"mocap_to_ros2_receive={mocap_to_ros2_receive_ms:.2f}] "
                f"packets={self.count}"
            )
            self.last_log_time = now


def main():
    parser = argparse.ArgumentParser(description="Receive SoftTouch ROS1 mocap UDP and publish ROS2 pose/twist.")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15551)
    parser.add_argument("--pose-topic", default="/softtouch/mocap/chest/pose")
    parser.add_argument("--twist-topic", default="/softtouch/mocap/chest/twist")
    parser.add_argument("--frame-id", default="mocap_world")
    parser.add_argument(
        "--position-offset-world",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="World-frame xyz offset added to each published pose, in metres.",
    )
    parser.add_argument(
        "--position-offset-body",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Body-frame xyz offset rotated by the mocap orientation then added to each published pose, in metres.",
    )
    parser.add_argument(
        "--max-position-step-m",
        type=float,
        default=0.0,
        help="Drop packets whose position jumps more than this within --jump-check-max-dt. 0 disables.",
    )
    parser.add_argument(
        "--max-orientation-step-deg",
        type=float,
        default=0.0,
        help="Drop packets whose orientation jumps more than this within --jump-check-max-dt. 0 disables.",
    )
    parser.add_argument(
        "--jump-check-max-dt",
        type=float,
        default=0.05,
        help="Only apply jump rejection to consecutive packets separated by at most this many seconds.",
    )
    args, ros_args = parser.parse_known_args()
    args.position_offset_world = tuple(float(value) for value in args.position_offset_world)
    args.position_offset_body = tuple(float(value) for value in args.position_offset_body)

    rclpy.init(args=ros_args)
    node = MocapUdpReceiver(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
