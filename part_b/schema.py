from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Optional, Tuple


GpsState = Literal["good", "degraded", "lost"]
LaneSide = Literal["left", "right", "center", "unknown"]


@dataclass
class GpsSample:
    lat: float
    lon: float
    alt: float
    speed_mps: float
    hdop: Optional[float]
    satellites: Optional[int]
    valid: bool = True


@dataclass
class ImuSample:
    ax: float
    ay: float
    az: float
    gx: float
    gy: float
    gz: float


@dataclass
class Pose2D:
    x: float
    y: float
    theta: float


@dataclass
class DeltaPose:
    """Frame-to-frame vehicle/body-frame motion.

    Contract:
    - dx: lateral motion in meters, positive to the vehicle/camera right.
    - dy: forward motion in meters, positive along the vehicle heading.
    - dtheta: yaw change in radians, positive counter-clockwise in local pose.

    `pose_local` and `fused_pose` are already accumulated in the local/world
    frame. Consumers must rotate dx/dy by the current heading exactly once.
    """

    dx: float
    dy: float
    dtheta: float
    scale: float
    matches: int
    inliers: int
    valid: bool


@dataclass
class LaneEstimate:
    lane_side: LaneSide
    lane_center_px: Optional[float]
    confidence: float
    left_count: int
    right_count: int


@dataclass
class EventEstimate:
    u_turn: bool
    heading_delta_deg: float


@dataclass
class HandoverEstimate:
    mode: str
    transition: Optional[str]
    lost_frames: int
    relock_frames: int
    gps_correction_noise_m: float
    latched_gps: Optional[GpsSample]
    latched_xy: Optional[Tuple[float, float]]
    latched_theta: Optional[float]
    loss_error_m: Optional[float]
    relock_error_m: Optional[float]


@dataclass
class Phase3Frame:
    sequence_id: str
    frame_index: int
    timestamp: float
    timestamp_iso: str
    frame_path: str
    gps: GpsSample
    imu: ImuSample


@dataclass
class Phase3Output:
    sample: Phase3Frame
    gps_local_xy: Optional[Tuple[float, float]]
    pose_local: Pose2D
    fused_pose: Pose2D
    delta_pose: DeltaPose
    lane: LaneEstimate
    gps_state: GpsState
    events: EventEstimate
    handover: Optional[HandoverEstimate] = None
    vo_scale_source: Optional[str] = None
    vo_scale_hint_m: Optional[float] = None

    def to_jsonable(self) -> Dict[str, Any]:
        return asdict(self)


def empty_delta() -> DeltaPose:
    return DeltaPose(
        dx=0.0,
        dy=0.0,
        dtheta=0.0,
        scale=0.0,
        matches=0,
        inliers=0,
        valid=False,
    )


def empty_pose() -> Pose2D:
    return Pose2D(x=0.0, y=0.0, theta=0.0)


def ground_delta_from_points(
    prev_xy: Optional[Tuple[float, float]],
    curr_xy: Optional[Tuple[float, float]],
) -> float:
    if prev_xy is None or curr_xy is None:
        return 0.0
    dx = curr_xy[0] - prev_xy[0]
    dy = curr_xy[1] - prev_xy[1]
    return float((dx * dx + dy * dy) ** 0.5)
