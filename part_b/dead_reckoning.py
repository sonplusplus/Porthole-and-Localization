import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .schema import DeltaPose, MotionSource, Pose2D, empty_delta


@dataclass
class WheelImuDeadReckoner:
    """Frame-to-frame wheel/IMU odometry for GPS-denied prediction.

    In KITTI Raw, ``speed_mps`` comes from OXTS velocity and is used here as a
    wheel-speed proxy. On a real vehicle this input should come from CAN wheel
    odometry instead.
    """

    max_dt_s: float = 1.0
    previous_timestamp: Optional[float] = None

    def update(self, timestamp: float, speed_mps: float, gyro_z_rad_s: float) -> DeltaPose:
        if not _finite(timestamp, speed_mps, gyro_z_rad_s):
            self.previous_timestamp = None
            return empty_delta()

        if self.previous_timestamp is None:
            self.previous_timestamp = float(timestamp)
            return empty_delta()

        dt = float(timestamp) - self.previous_timestamp
        self.previous_timestamp = float(timestamp)
        if dt <= 0.0 or dt > self.max_dt_s:
            return empty_delta()

        distance = float(speed_mps) * dt
        yaw = float(gyro_z_rad_s) * dt
        return DeltaPose(
            dx=0.0,
            dy=distance,
            dtheta=yaw,
            scale=abs(distance),
            matches=0,
            inliers=0,
            valid=True,
        )


def choose_motion_delta(
    motion_source: MotionSource,
    vo_delta: DeltaPose,
    wheel_imu_delta: DeltaPose,
) -> Tuple[DeltaPose, str]:
    if motion_source == "vo":
        return vo_delta, "vo" if vo_delta.valid else "invalid"

    if motion_source == "wheel_imu":
        return wheel_imu_delta, "wheel_imu" if wheel_imu_delta.valid else "invalid"

    if wheel_imu_delta.valid and vo_delta.valid:
        return _fuse_vo_wheel_imu(vo_delta, wheel_imu_delta), "vo_wheel_imu"
    if wheel_imu_delta.valid:
        return wheel_imu_delta, "wheel_imu"
    if vo_delta.valid:
        return vo_delta, "vo"
    return empty_delta(), "invalid"


def integrate_body_delta(pose: Pose2D, delta: DeltaPose) -> Pose2D:
    if not delta.valid:
        return pose

    c = math.cos(pose.theta)
    s = math.sin(pose.theta)
    dx_world = c * delta.dx - s * delta.dy
    dy_world = s * delta.dx + c * delta.dy
    return Pose2D(
        x=pose.x + dx_world,
        y=pose.y + dy_world,
        theta=_wrap_angle(pose.theta + delta.dtheta),
    )


def _fuse_vo_wheel_imu(vo_delta: DeltaPose, wheel_imu_delta: DeltaPose) -> DeltaPose:
    lateral_limit = abs(wheel_imu_delta.dy) * 0.25
    dx = _clamp(vo_delta.dx, -lateral_limit, lateral_limit)
    return DeltaPose(
        dx=dx,
        dy=wheel_imu_delta.dy,
        dtheta=wheel_imu_delta.dtheta,
        scale=wheel_imu_delta.scale,
        matches=vo_delta.matches,
        inliers=vo_delta.inliers,
        valid=True,
    )


def _finite(*values: float) -> bool:
    try:
        return all(math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        return False


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2 * math.pi) - math.pi)
