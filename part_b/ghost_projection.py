import math
from dataclasses import dataclass
from typing import Optional, Tuple

from .landmark_schema import LandmarkRecord
from .schema import Pose2D


@dataclass
class GhostProjection:
    landmark_id: str
    class_name: str
    center_xy: Tuple[float, float]
    bbox_xyxy: Tuple[int, int, int, int]
    distance_m: float
    bearing_rad: float


def project_landmark(
    record: LandmarkRecord,
    pose: Pose2D,
    camera_fx: float,
    camera_cx: float,
    image_width: int,
    image_height: int,
) -> Optional[GhostProjection]:
    dx = float(record.p_3D.x - pose.x)
    dy = float(record.p_3D.y - pose.y)
    distance = math.hypot(dx, dy)
    if distance <= 0.5:
        return None

    bearing = _wrap_angle(math.atan2(dy, dx) - pose.theta)
    if abs(bearing) > math.radians(75):
        return None

    u = float(camera_cx + camera_fx * math.tan(bearing))
    if u < 0 or u >= image_width:
        return None

    bbox = _last_bbox(record)
    if bbox is None:
        width = max(18, int(180.0 / max(distance, 1.0)))
        height = width
        v = int(image_height * 0.45)
        bbox = (
            int(u - width * 0.5),
            int(v - height * 0.5),
            int(u + width * 0.5),
            int(v + height * 0.5),
        )
    else:
        x1, y1, x2, y2 = bbox
        width = max(4, x2 - x1)
        height = max(4, y2 - y1)
        cx_old = 0.5 * (x1 + x2)
        shift = u - cx_old
        bbox = (
            int(round(x1 + shift)),
            int(round(y1)),
            int(round(x2 + shift)),
            int(round(y2)),
        )

    bbox = _clamp_bbox(bbox, image_width, image_height)
    center = (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3]))
    return GhostProjection(record.id, record.class_name, center, bbox, distance, bearing)


def reprojection_error_px(projection: GhostProjection, bbox_xyxy: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = bbox_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    dx = projection.center_xy[0] - cx
    dy = projection.center_xy[1] - cy
    return float(math.hypot(dx, dy))


def _last_bbox(record: LandmarkRecord) -> Optional[Tuple[int, int, int, int]]:
    value = record.attributes.get("last_bbox_xyxy")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    return tuple(int(v) for v in value)


def _clamp_bbox(bbox: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2 * math.pi) - math.pi)
