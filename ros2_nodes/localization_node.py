from __future__ import annotations


def _load_ros2():
    try:
        import rclpy
        from geometry_msgs.msg import PoseStamped
        from nav_msgs.msg import Odometry
        from rclpy.node import Node
    except ImportError:
        return None, None, None, None
    return rclpy, Node, Odometry, PoseStamped


def main() -> None:
    rclpy, Node, Odometry, PoseStamped = _load_ros2()
    if rclpy is None:
        print("ROS 2 is not installed. This stub documents /localization/odometry and /localization/pose publishers.")
        return

    class LocalizationNode(Node):
        def __init__(self) -> None:
            super().__init__("localization_node")
            self.odom_pub = self.create_publisher(Odometry, "/localization/odometry", 10)
            self.pose_pub = self.create_publisher(PoseStamped, "/localization/pose", 10)
            self.get_logger().info("localization_node ready: /localization/odometry, /localization/pose")

    rclpy.init()
    node = LocalizationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
