import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .schema import Pose2D
from .landmark_schema import LandmarkObservation, Point3D, VisualDescriptor, normalize_text
from .ocr import OcrBackend


@dataclass
class DetectorConfig:
    min_area_px: int = 350
    max_area_ratio: float = 0.08
    assumed_sign_distance_m: float = 18.0
    assumed_street_name_distance_m: float = 24.0
    street_aspect_min: float = 1.55
    descriptor_size: int = 32


class Phase4LandmarkDetector:
    """OpenCV baseline for sign/street-name landmark observations.

    This is a bootstrap detector. It produces candidate landmarks and stable
    descriptors without requiring pretrained OCR/sign models.
    """

    def __init__(
        self,
        camera_fx: float,
        camera_cx: float,
        ocr: OcrBackend,
        config: Optional[DetectorConfig] = None,
    ) -> None:
        self.camera_fx = camera_fx
        self.camera_cx = camera_cx
        self.ocr = ocr
        self.config = config or DetectorConfig()
        self.orb = cv2.ORB_create(nfeatures=96)

    def detect(
        self,
        frame: np.ndarray,
        pose: Pose2D,
        sequence_id: str,
        frame_index: int,
        timestamp: float,
        depth_metric: Optional[np.ndarray] = None,
    ) -> List[LandmarkObservation]:
        mask = _traffic_color_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        observations: List[LandmarkObservation] = []
        image_area = frame.shape[0] * frame.shape[1]
        for contour_index, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area < self.config.min_area_px or area > image_area * self.config.max_area_ratio:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            if w <= 8 or h <= 8:
                continue
            if (w * h) > image_area * self.config.max_area_ratio:
                continue

            bbox = _clamp_bbox((x, y, x + w, y + h), frame.shape[1], frame.shape[0])
            roi = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            if roi.size == 0:
                continue

            aspect = w / max(h, 1)
            shape = _classify_shape(contour)
            color = _dominant_sign_color(frame, contour)
            class_name = "street_name_sign" if aspect >= self.config.street_aspect_min and color == "blue" else "traffic_sign"

            ocr_result = self.ocr.recognize(roi) if class_name == "street_name_sign" else None
            descriptor = self._describe(roi, ocr_result.text if ocr_result else None)
            p_3d = self._estimate_position(bbox, pose, class_name, depth_metric)

            observations.append(
                LandmarkObservation(
                    observation_id=f"O_{sequence_id}_{frame_index:06d}_{contour_index:03d}",
                    class_name=class_name,
                    p_3D=p_3d,
                    d_visual=descriptor,
                    timestamp=timestamp,
                    frame_index=frame_index,
                    bbox_xyxy=bbox,
                    source="phase4_color_shape",
                    attributes={
                        "shape": shape,
                        "color": color,
                        "bbox_area_px": int(w * h),
                        "ocr_backend": self.ocr.name,
                        "ocr_confidence": ocr_result.confidence if ocr_result else None,
                    },
                )
            )
        return observations

    def _describe(self, roi: np.ndarray, text_raw: Optional[str]) -> VisualDescriptor:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        _, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(descriptors) == 0:
            vector = _image_fallback_descriptor(gray, self.config.descriptor_size)
            quality = 0.25
            kind = "intensity_grid"
        else:
            mean_descriptor = descriptors.astype(np.float32).mean(axis=0) / 255.0
            vector = mean_descriptor[: self.config.descriptor_size].astype(float).tolist()
            quality = min(1.0, len(descriptors) / 48.0)
            kind = "orb_mean"

        return VisualDescriptor(
            kind=kind,
            vector=vector,
            quality=quality,
            text_raw=text_raw,
            text_norm=normalize_text(text_raw),
        )

    def _estimate_position(
        self,
        bbox: Tuple[int, int, int, int],
        pose: Pose2D,
        class_name: str,
        depth_metric: Optional[np.ndarray] = None,
    ) -> Point3D:
        x1, y1, x2, y2 = bbox
        cx_px = (x1 + x2) * 0.5
        cy_px = (y1 + y2) * 0.5

        distance = None
        quality = 0.35
        if depth_metric is not None and depth_metric.ndim >= 2:
            h, w = depth_metric.shape[:2]
            if h > 0 and w > 0:
                margin = 3
                cy = int(round(cy_px))
                cx = int(round(cx_px))
                r0 = max(0, cy - margin)
                r1 = min(h, cy + margin + 1)
                c0 = max(0, cx - margin)
                c1 = min(w, cx + margin + 1)
                patch = depth_metric[r0:r1, c0:c1]
                if patch.size > 0:
                    values = patch[np.isfinite(patch)]
                    values = values[(values >= 3.0) & (values <= 60.0)]
                    if values.size > 0:
                        distance = float(np.median(values))
                        quality = 0.70

        if distance is None:
            distance = (
                self.config.assumed_street_name_distance_m
                if class_name == "street_name_sign"
                else self.config.assumed_sign_distance_m
            )

        bearing = math.atan2(cx_px - self.camera_cx, max(self.camera_fx, 1.0))
        heading = pose.theta + bearing
        return Point3D(
            x=pose.x + distance * math.cos(heading),
            y=pose.y + distance * math.sin(heading),
            z=0.0,
            quality=quality,
        )


def _traffic_color_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 70, 55), (12, 255, 255))
    red2 = cv2.inRange(hsv, (168, 70, 55), (180, 255, 255))
    blue = cv2.inRange(hsv, (88, 45, 45), (132, 255, 255))
    yellow = cv2.inRange(hsv, (17, 65, 70), (38, 255, 255))
    mask = cv2.bitwise_or(cv2.bitwise_or(red1, red2), cv2.bitwise_or(blue, yellow))
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def _dominant_sign_color(frame: np.ndarray, contour: np.ndarray) -> str:
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    pixels = hsv[mask > 0]
    if pixels.size == 0:
        return "unknown"
    hue = float(np.median(pixels[:, 0]))
    if hue <= 12 or hue >= 168:
        return "red"
    if 88 <= hue <= 132:
        return "blue"
    if 17 <= hue <= 38:
        return "yellow"
    return "unknown"


def _classify_shape(contour: np.ndarray) -> str:
    perimeter = cv2.arcLength(contour, True)
    if perimeter <= 0:
        return "unknown"
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    vertices = len(approx)
    if vertices == 3:
        return "triangle"
    if vertices == 4:
        return "rectangle"
    area = cv2.contourArea(contour)
    circularity = 4.0 * math.pi * area / max(perimeter * perimeter, 1e-6)
    if circularity > 0.62:
        return "circle"
    return "polygon"


def _clamp_bbox(
    bbox: Tuple[int, int, int, int],
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, x1)),
        max(0, min(height - 1, y1)),
        max(1, min(width, x2)),
        max(1, min(height, y2)),
    )


def _image_fallback_descriptor(gray: np.ndarray, size: int) -> List[float]:
    side = int(math.sqrt(size))
    side = max(side, 1)
    small = cv2.resize(gray, (side, side), interpolation=cv2.INTER_AREA).astype(np.float32)
    vector = (small.reshape(-1) / 255.0).tolist()
    if len(vector) < size:
        vector.extend([0.0] * (size - len(vector)))
    return vector[:size]
