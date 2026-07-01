#!/usr/bin/env python3
"""Broadcast world->pelvis TF from /softtouch/base/pose.

The SoftTouch MuJoCo bridge publishes the pelvis pose in the world frame as a
PoseStamped, and robot_state_publisher roots the robot at 'pelvis' with no
parent. So nothing puts 'world' into the TF tree -> RViz (Fixed Frame = world,
where the dribble route markers live) shows nothing. This node bridges that:
world -> pelvis from the live base pose, so RViz places the robot and draws it
moving along the route.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster


class BaseTfBroadcaster(Node):
    def __init__(self):
        super().__init__("softtouch_base_tf_broadcaster")
        self.declare_parameter("pose_topic", "/softtouch/base/pose")
        self.declare_parameter("parent_frame", "world")
        self.declare_parameter("child_frame", "pelvis")
        topic = self.get_parameter("pose_topic").value
        self.parent = self.get_parameter("parent_frame").value
        self.child = self.get_parameter("child_frame").value
        self.br = TransformBroadcaster(self)
        # bridge publishes base pose BEST_EFFORT (sensor-like); match it or we
        # get an incompatible-QoS warning and receive nothing.
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.BEST_EFFORT
        self.create_subscription(PoseStamped, topic, self.cb, qos)
        self.get_logger().info(
            f"broadcasting {self.parent}->{self.child} from {topic}")

    def cb(self, msg: PoseStamped):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.parent
        t.child_frame_id = self.child
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.br.sendTransform(t)


def main():
    rclpy.init()
    node = BaseTfBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
