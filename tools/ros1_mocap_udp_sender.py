#!/usr/bin/python3
import argparse
import socket
import struct
import time

import rospy
from hiperlab_rostools.msg import mocap_output


PACKET = struct.Struct("<4sBiqddddddddd")
MAGIC = b"STMC"
VERSION = 1


def stamp_to_sec(stamp):
    value = stamp.to_sec()
    return value if value > 0.0 else time.time()


def main():
    parser = argparse.ArgumentParser(description="Forward ROS1 hiperlab mocap_output to localhost UDP.")
    parser.add_argument("--topic", default="/mocap_output1")
    parser.add_argument("--vehicle-id", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=15551)
    parser.add_argument("--queue-size", type=int, default=1)
    args = parser.parse_args(rospy.myargv()[1:])

    rospy.init_node("softtouch_ros1_mocap_udp_sender", anonymous=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.IPPROTO_IP, socket.IP_TOS, 0x10)
    target = (args.host, args.port)
    sent = 0

    def callback(msg):
        nonlocal sent
        if args.vehicle_id >= 0 and msg.vehicleID != args.vehicle_id:
            return
        packet = PACKET.pack(
            MAGIC,
            VERSION,
            int(msg.vehicleID),
            int(msg.header.seq),
            stamp_to_sec(msg.header.stamp),
            float(msg.posx),
            float(msg.posy),
            float(msg.posz),
            float(msg.attq0),
            float(msg.attq1),
            float(msg.attq2),
            float(msg.attq3),
            time.time(),
        )
        sock.sendto(packet, target)
        sent += 1

    rospy.Subscriber(args.topic, mocap_output, callback, queue_size=args.queue_size, tcp_nodelay=True)
    rospy.loginfo(
        "SoftTouch ROS1 mocap UDP sender: %s vehicle_id=%s -> %s:%d",
        args.topic,
        args.vehicle_id,
        args.host,
        args.port,
    )
    rospy.spin()
    rospy.loginfo("SoftTouch ROS1 mocap UDP sender stopped after %d packets.", sent)


if __name__ == "__main__":
    main()
