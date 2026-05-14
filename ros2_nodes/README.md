# ROS 2 Replay Interface

These nodes replay the artifacts produced by the CPU pipelines so the project
has a concrete ROS 2 interface without requiring ROS 2 in the normal dev
workspace.

Topics:

- `/camera/image_raw` - `sensor_msgs/msg/Image`, optional video replay from `image_source`
- `/potholes` - `std_msgs/msg/String`, JSON payload grouped by `frame_index`
- `/localization/odometry` - `nav_msgs/msg/Odometry`, from Phase 3 `fused_pose`
- `/localization/pose` - `geometry_msgs/msg/PoseStamped`, from Phase 3 `fused_pose`
- `/localization/path` - `nav_msgs/msg/Path`, accumulated replay trajectory

Examples after building this folder in a ROS 2 workspace:

```bash
ros2 run ev_pothole_localization_nodes pothole_node \
  --ros-args \
  -p detections_jsonl:=data/phase2b_outputs/real_demos/vid1_detections.jsonl \
  -p image_source:=data/demo_inputs/vid1.mp4 \
  -p publish_rate_hz:=10.0
```

```bash
ros2 run ev_pothole_localization_nodes localization_node \
  --ros-args \
  -p phase3_jsonl:=data/phase3_outputs/phase5_handover_latest.jsonl \
  -p publish_rate_hz:=10.0
```

Or launch both replay publishers together:

```bash
ros2 launch ev_pothole_localization_nodes replay_demo.launch.py \
  detections_jsonl:=data/phase2b_outputs/real_demos/vid1_detections.jsonl \
  image_source:=data/demo_inputs/vid1.mp4 \
  phase3_jsonl:=data/phase3_outputs/phase5_handover_latest.jsonl \
  publish_rate_hz:=10.0
```

The Python files remain import-safe on machines without ROS 2 and print a clear
message when run outside a ROS 2 environment.
