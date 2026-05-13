from typing import Optional, Tuple

import cv2
import numpy as np

from .phase3_schema import LaneEstimate


class LaneDetectorBaseline:
    """Simple CPU lane-side baseline using Canny + Hough lines."""

    def __init__(self, center_deadband_px: float = 35.0) -> None:
        self.center_deadband_px = center_deadband_px

    def estimate(self, frame: np.ndarray) -> LaneEstimate:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 60, 160)

        mask = np.zeros_like(edges)
        roi = np.array(
            [[
                (int(0.08 * w), h),
                (int(0.45 * w), int(0.58 * h)),
                (int(0.55 * w), int(0.58 * h)),
                (int(0.92 * w), h),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, roi, 255)
        edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=45,
            minLineLength=max(30, w // 18),
            maxLineGap=60,
        )
        if lines is None:
            return LaneEstimate("unknown", None, 0.0, 0, 0)

        left_x = []
        right_x = []
        y_eval = h * 0.78
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.35:
                continue
            intercept = y1 - slope * x1
            x_eval = (y_eval - intercept) / slope
            if not (0 <= x_eval <= w):
                continue
            if slope < 0:
                left_x.append(float(x_eval))
            else:
                right_x.append(float(x_eval))

        if not left_x or not right_x:
            return LaneEstimate("unknown", None, 0.0, len(left_x), len(right_x))

        left = float(np.median(left_x))
        right = float(np.median(right_x))
        lane_center = 0.5 * (left + right)
        offset = (w * 0.5) - lane_center
        confidence = min(1.0, (len(left_x) + len(right_x)) / 12.0)

        if abs(offset) <= self.center_deadband_px:
            side = "center"
        elif offset < 0:
            side = "left"
        else:
            side = "right"
        return LaneEstimate(side, lane_center, confidence, len(left_x), len(right_x))
