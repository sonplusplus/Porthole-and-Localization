import math
from typing import Optional, Tuple

import numpy as np

from .phase3_schema import DeltaPose, GpsState, Pose2D


class LocalizationEKF:
    """Small EKF for Phase 3 baseline: predict with VO, correct x/y with GPS."""

    def __init__(self) -> None:
        self.x = np.zeros((3, 1), dtype=np.float64)
        self.p = np.diag([20.0, 20.0, 0.5]).astype(np.float64)
        self.initialized = False

    def update(
        self,
        delta: DeltaPose,
        gps_xy: Optional[Tuple[float, float]],
        gps_state: GpsState,
    ) -> Pose2D:
        if not self.initialized and gps_xy is not None and gps_state == "good":
            self.x[0, 0] = gps_xy[0]
            self.x[1, 0] = gps_xy[1]
            self.initialized = True

        self._predict(delta)
        if gps_xy is not None and gps_state == "good":
            self._correct_gps(gps_xy)

        return Pose2D(
            x=float(self.x[0, 0]),
            y=float(self.x[1, 0]),
            theta=float(self.x[2, 0]),
        )

    def _predict(self, delta: DeltaPose) -> None:
        if not delta.valid:
            self.p = self.p + np.diag([0.15, 0.15, 0.01])
            return

        theta = self.x[2, 0]
        c = math.cos(theta)
        s = math.sin(theta)
        dx_body = delta.dx
        dy_body = delta.dy

        self.x[0, 0] += c * dx_body - s * dy_body
        self.x[1, 0] += s * dx_body + c * dy_body
        self.x[2, 0] = _wrap_angle(self.x[2, 0] + delta.dtheta)

        f = np.array(
            [
                [1.0, 0.0, -s * dx_body - c * dy_body],
                [0.0, 1.0, c * dx_body - s * dy_body],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q_scale = max(delta.scale, 1.0)
        q = np.diag([0.05 * q_scale, 0.05 * q_scale, 0.01]).astype(np.float64)
        self.p = f @ self.p @ f.T + q

    def _correct_gps(self, gps_xy: Tuple[float, float]) -> None:
        z = np.array([[gps_xy[0]], [gps_xy[1]]], dtype=np.float64)
        h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        r = np.diag([3.0, 3.0]).astype(np.float64)
        y = z - h @ self.x
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.x[2, 0] = _wrap_angle(self.x[2, 0])
        self.p = (np.eye(3) - k @ h) @ self.p


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2 * math.pi) - math.pi)
