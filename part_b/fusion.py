import math
from typing import Optional, Tuple

import numpy as np

from .landmark_schema import Point3D
from .schema import DeltaPose, GpsState, Pose2D


class LocalizationEKF:
    """Small EKF for Phase 3 baseline: predict with VO, correct x/y with GPS.

    Noise parameters are configurable because real GPS modules vary widely.
    ``gps_noise_m`` is a standard deviation in metres; covariance uses its square.
    """

    def __init__(
        self,
        p0: Optional[np.ndarray] = None,
        q_xy: float = 0.05,
        q_theta: float = 0.01,
        q_invalid_xy: float = 0.15,
        q_invalid_theta: float = 0.01,
        r_gps_noise_m: float = 3.0,
    ) -> None:
        self.x = np.zeros((3, 1), dtype=np.float64)
        self.p = np.array(p0, dtype=np.float64, copy=True) if p0 is not None else np.diag([20.0, 20.0, 0.5]).astype(np.float64)
        self.q_xy = float(q_xy)
        self.q_theta = float(q_theta)
        self.q_invalid_xy = float(q_invalid_xy)
        self.q_invalid_theta = float(q_invalid_theta)
        self.r_gps_noise_m = float(r_gps_noise_m)
        self.initialized = False

    def update(
        self,
        delta: DeltaPose,
        gps_xy: Optional[Tuple[float, float]],
        gps_state: GpsState,
        gps_noise_m: Optional[float] = None,
    ) -> Pose2D:
        if not self.initialized and gps_xy is not None and gps_state == "good":
            self.x[0, 0] = gps_xy[0]
            self.x[1, 0] = gps_xy[1]
            self.initialized = True

        self._predict(delta)
        if gps_xy is not None and gps_state == "good":
            self._correct_gps(gps_xy, gps_noise_m=gps_noise_m)

        return self.pose()

    def pose(self) -> Pose2D:
        return Pose2D(
            x=float(self.x[0, 0]),
            y=float(self.x[1, 0]),
            theta=float(self.x[2, 0]),
        )

    def correct_landmark(
        self,
        observed_landmark: Point3D,
        reference_landmark: Point3D,
        noise_m: Optional[float] = None,
        match_score: float = 1.0,
    ) -> Pose2D:
        """Correct vehicle x/y from a matched landmark residual.

        ``observed_landmark`` is the world position implied by the current pose
        and the image observation. ``reference_landmark`` is the stored DB
        position. Their delta is an approximate vehicle translation residual.
        """

        dx = float(reference_landmark.x - observed_landmark.x)
        dy = float(reference_landmark.y - observed_landmark.y)
        measurement_xy = (float(self.x[0, 0] + dx), float(self.x[1, 0] + dy))

        quality = max(0.1, min(1.0, float(observed_landmark.quality or 0.35)))
        score = max(0.1, min(1.0, float(match_score)))
        noise = float(noise_m) if noise_m is not None else 4.0 / (quality * score)
        self._correct_xy(measurement_xy, noise_m=noise)
        return self.pose()

    def _predict(self, delta: DeltaPose) -> None:
        if not delta.valid:
            self.p = self.p + np.diag([self.q_invalid_xy, self.q_invalid_xy, self.q_invalid_theta])
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
        q = np.diag([self.q_xy * q_scale, self.q_xy * q_scale, self.q_theta]).astype(np.float64)
        self.p = f @ self.p @ f.T + q

    def _correct_gps(self, gps_xy: Tuple[float, float], gps_noise_m: Optional[float] = None) -> None:
        noise = self.r_gps_noise_m if gps_noise_m is None else float(gps_noise_m)
        self._correct_xy(gps_xy, noise_m=noise)

    def _correct_xy(self, xy: Tuple[float, float], noise_m: float) -> None:
        z = np.array([[xy[0]], [xy[1]]], dtype=np.float64)
        h = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        noise = max(float(noise_m), 0.1)
        r = np.diag([noise * noise, noise * noise]).astype(np.float64)
        y = z - h @ self.x
        s = h @ self.p @ h.T + r
        k = self.p @ h.T @ np.linalg.inv(s)
        self.x = self.x + k @ y
        self.x[2, 0] = _wrap_angle(self.x[2, 0])
        self.p = (np.eye(3) - k @ h) @ self.p


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2 * math.pi) - math.pi)
