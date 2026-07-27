#!/usr/bin/python3
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


class MocapPoseViz(Node):
    def __init__(self):
        super().__init__("softtouch_mocap_pose_viz")
        self.declare_parameter("pose_topic", "/softtouch/mocap/chest/pose")
        self.declare_parameter("child_frame_id", "softtouch_mocap_chest")
        self.declare_parameter("marker_namespace", "softtouch_mocap_chest")
        self.declare_parameter("marker_topic", "/softtouch/mocap/chest/markers")
        self.declare_parameter("path_topic", "/softtouch/mocap/chest/path")
        self.declare_parameter("path_max_points", 800)
        self.declare_parameter("publish_tf", True)
        self.declare_parameter("show_arrow", True)
        self.declare_parameter("sphere_diameter", 0.12)
        self.declare_parameter("sphere_color_r", 0.1)
        self.declare_parameter("sphere_color_g", 0.8)
        self.declare_parameter("sphere_color_b", 1.0)
        self.declare_parameter("sphere_color_a", 0.95)

        self.pose_topic = self.get_parameter("pose_topic").value
        self.child_frame_id = self.get_parameter("child_frame_id").value
        self.marker_namespace = self.get_parameter("marker_namespace").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.path_topic = self.get_parameter("path_topic").value
        self.path_max_points = int(self.get_parameter("path_max_points").value)
        self.publish_tf = bool(self.get_parameter("publish_tf").value)
        self.show_arrow = bool(self.get_parameter("show_arrow").value)
        self.sphere_diameter = float(self.get_parameter("sphere_diameter").value)
        self.sphere_color = (
            float(self.get_parameter("sphere_color_r").value),
            float(self.get_parameter("sphere_color_g").value),
            float(self.get_parameter("sphere_color_b").value),
            float(self.get_parameter("sphere_color_a").value),
        )

        self.tf_broadcaster = TransformBroadcaster(self) if self.publish_tf else None
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 1)
        self.path_pub = self.create_publisher(Path, self.path_topic, 1)
        self.path = Path()
        self.last_log_time = self.get_clock().now()

        self.sub = self.create_subscription(
            PoseStamped,
            self.pose_topic,
            self.pose_callback,
            1,
        )
        self.get_logger().info(
            f"Visualizing {self.pose_topic}: TF child={self.child_frame_id}, "
            f"markers={self.marker_topic}, path={self.path_topic}"
        )

    def pose_callback(self, msg):
        frame_id = msg.header.frame_id or "mocap_world"
        stamp = msg.header.stamp

        if self.tf_broadcaster is not None:
            tf_msg = TransformStamped()
            tf_msg.header.stamp = stamp
            tf_msg.header.frame_id = frame_id
            tf_msg.child_frame_id = self.child_frame_id
            tf_msg.transform.translation.x = msg.pose.position.x
            tf_msg.transform.translation.y = msg.pose.position.y
            tf_msg.transform.translation.z = msg.pose.position.z
            tf_msg.transform.rotation = msg.pose.orientation
            self.tf_broadcaster.sendTransform(tf_msg)

        self.path.header.stamp = stamp
        self.path.header.frame_id = frame_id
        self.path.poses.append(msg)
        if len(self.path.poses) > self.path_max_points:
            self.path.poses = self.path.poses[-self.path_max_points :]
        self.path_pub.publish(self.path)

        markers = MarkerArray()
        sphere = Marker()
        sphere.header = msg.header
        sphere.ns = self.marker_namespace
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose = msg.pose
        sphere.scale.x = self.sphere_diameter
        sphere.scale.y = self.sphere_diameter
        sphere.scale.z = self.sphere_diameter
        sphere.color.r = self.sphere_color[0]
        sphere.color.g = self.sphere_color[1]
        sphere.color.b = self.sphere_color[2]
        sphere.color.a = self.sphere_color[3]
        markers.markers.append(sphere)

        if self.show_arrow:
            arrow = Marker()
            arrow.header = msg.header
            arrow.ns = self.marker_namespace
            arrow.id = 1
            arrow.type = Marker.ARROW
            arrow.action = Marker.ADD
            arrow.pose = msg.pose
            arrow.scale.x = 0.45
            arrow.scale.y = 0.05
            arrow.scale.z = 0.05
            arrow.color.r = 0.2
            arrow.color.g = 1.0
            arrow.color.b = 0.25
            arrow.color.a = 0.95
            markers.markers.append(arrow)

        self.marker_pub.publish(markers)

        now = self.get_clock().now()
        if (now - self.last_log_time).nanoseconds > 2_000_000_000:
            q = msg.pose.orientation
            yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            self.get_logger().info(
                f"{self.marker_namespace} pos=({msg.pose.position.x:.3f}, {msg.pose.position.y:.3f}, "
                f"{msg.pose.position.z:.3f}) yaw={yaw:.3f}"
            )
            self.last_log_time = now


def main():
    rclpy.init()
    node = MocapPoseViz()
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
