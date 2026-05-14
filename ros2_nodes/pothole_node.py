from __future__ import annotations


def _load_ros2():
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        try:
            from pothole_msgs.msg import PotholeArray
        except ImportError:
            from std_msgs.msg import String as PotholeArray
    except ImportError:
        return None, None, None, None
    return rclpy, Node, Image, PotholeArray


def main() -> None:
    rclpy, Node, Image, PotholeArray = _load_ros2()
    if rclpy is None:
        print("ROS 2 is not installed. This stub documents /camera/image_raw and /potholes publishers.")
        return

    class PotholeNode(Node):
        def __init__(self) -> None:
            super().__init__("pothole_node")
            self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
            # Uses std_msgs/String as a fallback until custom PotholeArray is generated.
            self.pothole_pub = self.create_publisher(PotholeArray, "/potholes", 10)
            self.get_logger().info("pothole_node ready: /camera/image_raw, /potholes")

    rclpy.init()
    node = PotholeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
