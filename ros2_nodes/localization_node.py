from __future__ import annotations

try:
    from jsonl_replay import JsonlCursor, nested_get, pose_from_phase3_row, yaw_to_quaternion
except ImportError:
    from .jsonl_replay import JsonlCursor, nested_get, pose_from_phase3_row, yaw_to_quaternion


def _load_ros2():
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry, Path
        from rclpy.node import Node
    except ImportError:
        return None, None, None, None, None
    return rclpy, Node, Odometry, PoseStamped, Path


def main() -> None:
    rclpy, Node, Odometry, PoseStamped, Path = _load_ros2()
    if rclpy is None:
        print(
            "ROS 2 is not installed. This node replays Part B Phase 3 JSONL to "
            "/localization/odometry, /localization/pose, and /localization/path."
        )
        return

    class LocalizationNode(Node):
        def __init__(self) -> None:
            super().__init__("localization_node")
            self.declare_parameter("phase3_jsonl", "")
            self.declare_parameter("publish_rate_hz", 10.0)
            self.declare_parameter("loop", False)
            self.declare_parameter("frame_id", "map")
            self.declare_parameter("child_frame_id", "base_link")
            self.declare_parameter("odom_topic", "/localization/odometry")
            self.declare_parameter("pose_topic", "/localization/pose")
            self.declare_parameter("path_topic", "/localization/path")

            phase3_jsonl = str(self.get_parameter("phase3_jsonl").value or "")
            publish_rate_hz = max(0.1, float(self.get_parameter("publish_rate_hz").value))
            self.publish_dt = 1.0 / publish_rate_hz
            self.loop = bool(self.get_parameter("loop").value)
            self.frame_id = str(self.get_parameter("frame_id").value)
            self.child_frame_id = str(self.get_parameter("child_frame_id").value)
            odom_topic = str(self.get_parameter("odom_topic").value)
            pose_topic = str(self.get_parameter("pose_topic").value)
            path_topic = str(self.get_parameter("path_topic").value)

            self.odom_pub = self.create_publisher(Odometry, odom_topic, 10)
            self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
            self.path_pub = self.create_publisher(Path, path_topic, 10)
            self.cursor = JsonlCursor.from_path(phase3_jsonl, loop=self.loop) if phase3_jsonl else None
            self.path_msg = Path()
            self.path_msg.header.frame_id = self.frame_id
            self.prev_sample_timestamp = None

            if self.cursor is None:
                self.get_logger().warn("No phase3_jsonl configured; node will idle.")

            self.timer = self.create_timer(1.0 / publish_rate_hz, self._tick)
            self.get_logger().info(
                f"localization_node ready: {odom_topic} nav_msgs/Odometry, "
                f"{pose_topic} geometry_msgs/PoseStamped, {path_topic} nav_msgs/Path"
            )

        def _tick(self) -> None:
            if self.cursor is None:
                return
            row = self.cursor.next()
            if row is None:
                self.cursor = None
                self.get_logger().info("phase3_jsonl exhausted")
                return

            stamp = self.get_clock().now().to_msg()
            pose_msg = self._pose_msg(row, stamp)
            odom_msg = self._odom_msg(row, pose_msg)
            self.odom_pub.publish(odom_msg)
            self.pose_pub.publish(pose_msg)
            self.path_msg.header.stamp = stamp
            self.path_msg.poses.append(pose_msg)
            self.path_pub.publish(self.path_msg)

        def _pose_msg(self, row, stamp):
            x, y, theta = pose_from_phase3_row(row)
            qx, qy, qz, qw = yaw_to_quaternion(theta)
            msg = PoseStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = self.frame_id
            msg.pose.position.x = x
            msg.pose.position.y = y
            msg.pose.position.z = 0.0
            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz
            msg.pose.orientation.w = qw
            return msg

        def _odom_msg(self, row, pose_msg):
            msg = Odometry()
            msg.header = pose_msg.header
            msg.child_frame_id = self.child_frame_id
            msg.pose.pose = pose_msg.pose

            delta = row.get("delta_pose") if isinstance(row.get("delta_pose"), dict) else {}
            dt = self._row_dt(row)
            if dt > 0.0:
                msg.twist.twist.linear.x = float(delta.get("dy", 0.0)) / dt
                msg.twist.twist.linear.y = float(delta.get("dx", 0.0)) / dt
                msg.twist.twist.angular.z = float(delta.get("dtheta", 0.0)) / dt
            return msg

        def _row_dt(self, row) -> float:
            timestamp = nested_get(row, ("sample", "timestamp"))
            if timestamp is None:
                return self.publish_dt
            try:
                timestamp = float(timestamp)
            except (TypeError, ValueError):
                return self.publish_dt
            if self.prev_sample_timestamp is None:
                self.prev_sample_timestamp = timestamp
                return self.publish_dt
            dt = max(0.0, timestamp - self.prev_sample_timestamp)
            self.prev_sample_timestamp = timestamp
            return dt or self.publish_dt

    rclpy.init()
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
