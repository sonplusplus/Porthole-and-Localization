from __future__ import annotations

import json
from typing import Optional

try:
    from jsonl_replay import GroupedFrameCursor
except ImportError:
    from .jsonl_replay import GroupedFrameCursor


def _load_ros2():
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import Image
        from std_msgs.msg import String
    except ImportError:
        return None, None, None, None
    return rclpy, Node, Image, String


def _open_video(source: str):
    if not source:
        return None
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required when image_source is set") from exc

    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open image_source: {source}")
    return cap


def _image_msg_from_frame(frame, stamp, frame_id: str, Image):
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(frame.shape[0])
    msg.width = int(frame.shape[1])
    msg.encoding = "bgr8"
    msg.is_bigendian = 0
    msg.step = int(frame.shape[1] * frame.shape[2])
    msg.data = frame.tobytes()
    return msg


def main() -> None:
    rclpy, Node, Image, String = _load_ros2()
    if rclpy is None:
        print(
            "ROS 2 is not installed. This node replays Part A detection JSONL to "
            "/potholes as std_msgs/String and can optionally publish video frames "
            "to /camera/image_raw."
        )
        return

    class PotholeNode(Node):
        def __init__(self) -> None:
            super().__init__("pothole_node")
            self.declare_parameter("detections_jsonl", "")
            self.declare_parameter("image_source", "")
            self.declare_parameter("publish_rate_hz", 10.0)
            self.declare_parameter("loop", False)
            self.declare_parameter("frame_id", "camera")
            self.declare_parameter("potholes_topic", "/potholes")
            self.declare_parameter("image_topic", "/camera/image_raw")

            detections_jsonl = str(self.get_parameter("detections_jsonl").value or "")
            image_source = str(self.get_parameter("image_source").value or "")
            publish_rate_hz = max(0.1, float(self.get_parameter("publish_rate_hz").value))
            self.loop = bool(self.get_parameter("loop").value)
            self.frame_id = str(self.get_parameter("frame_id").value)
            potholes_topic = str(self.get_parameter("potholes_topic").value)
            image_topic = str(self.get_parameter("image_topic").value)

            self.image_pub = self.create_publisher(Image, image_topic, 10)
            self.pothole_pub = self.create_publisher(String, potholes_topic, 10)
            self.detection_cursor: Optional[GroupedFrameCursor] = None
            self.cap = None

            if detections_jsonl:
                self.detection_cursor = GroupedFrameCursor.from_detection_jsonl(detections_jsonl, loop=self.loop)
            if image_source:
                self.cap = _open_video(image_source)

            if self.detection_cursor is None and self.cap is None:
                self.get_logger().warn("No detections_jsonl or image_source configured; node will idle.")

            self.timer = self.create_timer(1.0 / publish_rate_hz, self._tick)
            self.get_logger().info(
                f"pothole_node ready: {image_topic} sensor_msgs/Image, {potholes_topic} std_msgs/String"
            )

        def _tick(self) -> None:
            stamp = self.get_clock().now().to_msg()
            self._publish_frame(stamp)
            self._publish_detections(stamp)

        def _publish_frame(self, stamp) -> None:
            if self.cap is None:
                return
            ok, frame = self.cap.read()
            if not ok:
                if not self.loop:
                    self.cap.release()
                    self.cap = None
                    self.get_logger().info("image_source exhausted")
                    return
                self.cap.set(1, 0)
                ok, frame = self.cap.read()
                if not ok:
                    return
            self.image_pub.publish(_image_msg_from_frame(frame, stamp, self.frame_id, Image))

        def _publish_detections(self, stamp) -> None:
            if self.detection_cursor is None:
                return
            payload = self.detection_cursor.next()
            if payload is None:
                self.detection_cursor = None
                self.get_logger().info("detections_jsonl exhausted")
                return
            payload["stamp"] = {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}
            payload["frame_id"] = self.frame_id
            msg = String()
            msg.data = json.dumps(payload, ensure_ascii=False)
            self.pothole_pub.publish(msg)

    rclpy.init()
    node = PotholeNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
