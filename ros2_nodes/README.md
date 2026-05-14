# ROS 2 Interface Stubs

These files document the intended ROS 2 deployment interface without requiring
ROS 2 to be installed in this workspace.

Topics:

- `/camera/image_raw` - `sensor_msgs/msg/Image`
- `/potholes` - custom `PotholeArray` message, one entry per detected pothole
- `/localization/odometry` - `nav_msgs/msg/Odometry`
- `/localization/pose` - `geometry_msgs/msg/PoseStamped`

The Python files are import-safe on machines without ROS 2 and print a clear
message when run outside a ROS 2 environment.
