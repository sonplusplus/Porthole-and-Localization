from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    detections_jsonl = LaunchConfiguration("detections_jsonl")
    image_source = LaunchConfiguration("image_source")
    phase3_jsonl = LaunchConfiguration("phase3_jsonl")
    publish_rate_hz = LaunchConfiguration("publish_rate_hz")
    loop = LaunchConfiguration("loop")

    return LaunchDescription(
        [
            DeclareLaunchArgument("detections_jsonl", default_value=""),
            DeclareLaunchArgument("image_source", default_value=""),
            DeclareLaunchArgument("phase3_jsonl", default_value=""),
            DeclareLaunchArgument("publish_rate_hz", default_value="10.0"),
            DeclareLaunchArgument("loop", default_value="false"),
            Node(
                package="ev_pothole_localization_nodes",
                executable="pothole_node",
                name="pothole_node",
                output="screen",
                parameters=[
                    {
                        "detections_jsonl": detections_jsonl,
                        "image_source": image_source,
                        "publish_rate_hz": publish_rate_hz,
                        "loop": loop,
                    }
                ],
            ),
            Node(
                package="ev_pothole_localization_nodes",
                executable="localization_node",
                name="localization_node",
                output="screen",
                parameters=[
                    {
                        "phase3_jsonl": phase3_jsonl,
                        "publish_rate_hz": publish_rate_hz,
                        "loop": loop,
                    }
                ],
            ),
        ]
    )
