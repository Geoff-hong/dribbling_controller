#!/usr/bin/python3
import argparse
import math
import socket
import struct
import time

import rospy
from geometry_msgs.msg import PointStamped


PACKET = struct.Struct("<4sBiqddddddddd")
MAGIC = b"STMC"
VERSION = 1


def stamp_to_sec(stamp):
    value = stamp.to_sec()
    return value if value > 0.0 else time.time()


def norm3(values):
    return math.sqrt(values[0] * values[0] + values[1] * values[1] + values[2] * values[2])


def sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def add3(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def mul3(a, scale):
    return (a[0] * scale, a[1] * scale, a[2] * scale)


class UnlabeledBallTracker:
    def __init__(self, args):
        self.args = args
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0x10)
        self.target = (args.host, args.port)

        self.group = []
        self.last_marker_stamp = None
        self.last_marker_wall = 0.0

        self.filtered_pos = None
        self.filtered_vel = (0.0, 0.0, 0.0)
        self.last_track_stamp = None
        self.last_measurement_stamp = None
        self.misses = 0
        self.sent = 0
        self.last_log_time = 0.0

        self.debug_pub = None
        if args.debug_topic:
            self.debug_pub = rospy.Publisher(args.debug_topic, PointStamped, queue_size=1)

        self.sub = rospy.Subscriber(
            args.topic,
            PointStamped,
            self.callback,
            queue_size=args.queue_size,
            tcp_nodelay=True,
        )
        self.timer = rospy.Timer(rospy.Duration(args.flush_timeout), self.flush_if_idle)
        rospy.loginfo(
            "SoftTouch ROS1 unlabeled ball tracker: %s -> %s:%d, expected_z=%.3f, z_range=[%.3f, %.3f]",
            args.topic,
            args.host,
            args.port,
            args.expected_z,
            args.min_z,
            args.max_z,
        )

    def callback(self, msg):
        stamp = stamp_to_sec(msg.header.stamp)
        if (
            self.group
            and self.last_marker_stamp is not None
            and stamp - self.last_marker_stamp > self.args.frame_gap
        ):
            self.process_group()

        self.group.append(msg)
        self.last_marker_stamp = stamp
        self.last_marker_wall = time.time()

    def flush_if_idle(self, _event):
        if self.group and time.time() - self.last_marker_wall > self.args.frame_gap:
            self.process_group()

    def point_tuple(self, msg):
        return (float(msg.point.x), float(msg.point.y), float(msg.point.z))

    def is_candidate(self, pos):
        x, y, z = pos
        if not all(math.isfinite(v) for v in pos):
            return False
        if x < self.args.min_x or x > self.args.max_x:
            return False
        if y < self.args.min_y or y > self.args.max_y:
            return False
        if z < self.args.min_z or z > self.args.max_z:
            return False
        return True

    def predicted_pos(self, stamp):
        if self.filtered_pos is None or self.last_track_stamp is None:
            return self.args.initial_position
        dt = max(0.0, stamp - self.last_track_stamp)
        return add3(self.filtered_pos, mul3(self.filtered_vel, dt))

    def measurement_is_fresh(self, stamp):
        return (
            self.last_measurement_stamp is not None
            and stamp - self.last_measurement_stamp <= self.args.max_hold_time
        )

    def choose_candidate(self, candidates, stamp):
        if not candidates:
            return None

        predicted = self.predicted_pos(stamp) if self.measurement_is_fresh(stamp) else self.args.initial_position
        best = None
        best_score = float("inf")
        for msg, pos in candidates:
            z_score = abs(pos[2] - self.args.expected_z) * self.args.z_weight
            if predicted is None:
                score = z_score
            else:
                d = norm3(sub3(pos, predicted))
                dt = 0.0
                if self.last_track_stamp is not None:
                    dt = max(0.0, stamp - self.last_track_stamp)
                gate = max(self.args.min_gate, self.args.max_speed * dt + self.args.gate_margin)
                if self.filtered_pos is not None and self.measurement_is_fresh(stamp) and d > gate:
                    continue
                score = d + z_score
            if score < best_score:
                best = (msg, pos)
                best_score = score
        return best

    def process_group(self):
        group = self.group
        self.group = []
        if not group:
            return

        stamp = max(stamp_to_sec(msg.header.stamp) for msg in group)
        total_count = len(group)
        candidates = [(msg, self.point_tuple(msg)) for msg in group]
        candidates = [(msg, pos) for msg, pos in candidates if self.is_candidate(pos)]
        was_stale = self.filtered_pos is not None and not self.measurement_is_fresh(stamp)
        chosen = self.choose_candidate(candidates, stamp)
        if chosen is None:
            self.handle_miss(group[-1], stamp, len(candidates), total_count)
            return

        msg, measured = chosen
        if self.filtered_pos is None or self.last_track_stamp is None:
            self.filtered_pos = measured
            self.filtered_vel = (0.0, 0.0, 0.0)
        else:
            dt = stamp - self.last_track_stamp
            if dt <= self.args.min_dt or dt > self.args.max_dt:
                self.filtered_pos = measured
                self.filtered_vel = (0.0, 0.0, 0.0)
            else:
                predicted = add3(self.filtered_pos, mul3(self.filtered_vel, dt))
                residual = sub3(measured, predicted)
                self.filtered_pos = add3(predicted, mul3(residual, self.args.alpha))
                self.filtered_vel = add3(self.filtered_vel, mul3(residual, self.args.beta / dt))
                speed = norm3(self.filtered_vel)
                if speed > self.args.max_speed:
                    self.filtered_vel = mul3(self.filtered_vel, self.args.max_speed / speed)

        self.last_track_stamp = stamp
        self.last_measurement_stamp = stamp
        self.misses = 0
        self.publish_ball(msg, stamp, self.filtered_pos)
        self.log_status("reacquire" if was_stale else "track", measured, self.filtered_pos, len(candidates), total_count, stamp)

    def handle_miss(self, source_msg, stamp, candidate_count, total_count):
        self.misses += 1
        if self.filtered_pos is None or self.last_track_stamp is None or self.last_measurement_stamp is None:
            self.log_status("lost", None, None, candidate_count, total_count, stamp)
            return

        dt = max(0.0, stamp - self.last_track_stamp)
        occlusion_time = max(0.0, stamp - self.last_measurement_stamp)
        self.filtered_vel = self.decay_velocity(self.filtered_vel, dt)

        if occlusion_time <= self.args.max_predict_time:
            predicted = add3(self.filtered_pos, mul3(self.filtered_vel, dt))
            self.filtered_pos = self.clamp_position(predicted)
            mode = "predict"
        else:
            if norm3(self.filtered_vel) < self.args.zero_velocity_epsilon:
                self.filtered_vel = (0.0, 0.0, 0.0)
            if occlusion_time <= self.args.max_hold_time:
                mode = "hold"
            else:
                mode = "stale_lost"
                self.log_status(mode, None, self.filtered_pos, candidate_count, total_count, stamp)
                if not self.args.publish_stale_hold:
                    return

        self.last_track_stamp = stamp
        self.publish_ball(source_msg, stamp, self.filtered_pos)
        self.log_status(mode, None, self.filtered_pos, candidate_count, total_count, stamp)

    def decay_velocity(self, vel, dt):
        if dt <= 0.0 or self.args.velocity_decay_time <= 0.0:
            return vel
        decay = math.exp(-dt / self.args.velocity_decay_time)
        return mul3(vel, decay)

    def clamp_position(self, pos):
        return (
            min(self.args.max_x, max(self.args.min_x, pos[0])),
            min(self.args.max_y, max(self.args.min_y, pos[1])),
            min(self.args.max_z, max(self.args.min_z, pos[2])),
        )

    def publish_ball(self, source_msg, stamp, pos):
        if pos is None or not all(math.isfinite(v) for v in pos):
            rospy.logerr("refusing to publish invalid ball position: %s", pos)
            return
        pos = self.clamp_position(pos)
        packet = PACKET.pack(
            MAGIC,
            VERSION,
            int(self.args.ball_id),
            int(source_msg.header.seq),
            float(stamp),
            float(pos[0]),
            float(pos[1]),
            float(pos[2]),
            1.0,
            0.0,
            0.0,
            0.0,
            time.time(),
        )
        self.sock.sendto(packet, self.target)
        self.sent += 1

        if self.debug_pub is not None:
            debug = PointStamped()
            debug.header = source_msg.header
            debug.header.stamp = rospy.Time.from_sec(stamp)
            debug.header.frame_id = self.args.debug_frame_id or source_msg.header.frame_id
            debug.point.x = pos[0]
            debug.point.y = pos[1]
            debug.point.z = pos[2]
            self.debug_pub.publish(debug)

    def log_status(self, mode, measured, filtered, candidate_count, total_count, stamp):
        now = time.time()
        if now - self.last_log_time < self.args.log_period:
            return
        self.last_log_time = now
        if filtered is None:
            rospy.logwarn(
                "ball %s: candidates=%d misses=%d grouped_markers=%d",
                mode,
                candidate_count,
                self.misses,
                total_count,
            )
            return
        age_ms = (now - stamp) * 1000.0
        if measured is None:
            measured = filtered
        rospy.loginfo(
            "ball mode=%s raw=(%.3f, %.3f, %.3f) filtered=(%.3f, %.3f, %.3f) "
            "vel=(%.2f, %.2f, %.2f) misses=%d candidates=%d/%d age=%.2f ms sent=%d",
            mode,
            measured[0],
            measured[1],
            measured[2],
            filtered[0],
            filtered[1],
            filtered[2],
            self.filtered_vel[0],
            self.filtered_vel[1],
            self.filtered_vel[2],
            self.misses,
            candidate_count,
            total_count,
            age_ms,
            self.sent,
        )


def build_parser():
    parser = argparse.ArgumentParser(description="Track one reflective soccer ball from ROS1 unlabeled mocap markers.")
    parser.add_argument("--topic", default="/mocap_unlabeled_marker")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15552)
    parser.add_argument("--ball-id", type=int, default=100)
    parser.add_argument("--queue-size", type=int, default=200)
    parser.add_argument("--frame-gap", type=float, default=0.002)
    parser.add_argument("--flush-timeout", type=float, default=0.01)
    parser.add_argument("--expected-z", type=float, default=0.10)
    parser.add_argument("--min-z", type=float, default=0.04)
    parser.add_argument("--max-z", type=float, default=0.35)
    parser.add_argument("--min-x", type=float, default=-5.0)
    parser.add_argument("--max-x", type=float, default=5.0)
    parser.add_argument("--min-y", type=float, default=-5.0)
    parser.add_argument("--max-y", type=float, default=5.0)
    parser.add_argument("--initial-position", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--alpha", type=float, default=0.75, help="Position correction gain for alpha-beta filter.")
    parser.add_argument("--beta", type=float, default=0.08, help="Velocity correction gain for alpha-beta filter.")
    parser.add_argument("--z-weight", type=float, default=0.25, help="Selection penalty per meter away from expected_z.")
    parser.add_argument("--min-gate", type=float, default=0.25)
    parser.add_argument("--gate-margin", type=float, default=0.08)
    parser.add_argument("--max-speed", type=float, default=8.0)
    parser.add_argument("--max-misses", type=int, default=20)
    parser.add_argument("--max-predict-time", type=float, default=0.20)
    parser.add_argument("--max-hold-time", type=float, default=1.00)
    parser.add_argument("--publish-stale-hold", action="store_true")
    parser.add_argument("--velocity-decay-time", type=float, default=0.20)
    parser.add_argument("--zero-velocity-epsilon", type=float, default=0.03)
    parser.add_argument("--min-dt", type=float, default=1.0e-4)
    parser.add_argument("--max-dt", type=float, default=0.1)
    parser.add_argument("--debug-topic", default="/softtouch/mocap/ball/point_ros1")
    parser.add_argument("--debug-frame-id", default="mocap_world")
    parser.add_argument("--log-period", type=float, default=1.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args(rospy.myargv()[1:])
    args.alpha = max(0.0, min(1.0, args.alpha))
    args.beta = max(0.0, min(1.0, args.beta))
    if args.initial_position is not None:
        args.initial_position = tuple(float(v) for v in args.initial_position)

    rospy.init_node("softtouch_ros1_unlabeled_ball_udp_sender", anonymous=True)
    UnlabeledBallTracker(args)
    rospy.spin()


if __name__ == "__main__":
    main()
