import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple

from .schema import GpsSample, GpsState, HandoverEstimate, Pose2D


@dataclass
class GpsLossSimulator:
    """Deterministic GPS dropout simulator for Phase 5 handover testing."""

    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    degraded_frames: int = 5
    degraded_hdop: float = 8.0
    degraded_satellites: int = 3

    @property
    def enabled(self) -> bool:
        return self.start_frame is not None and self.end_frame is not None

    def apply(self, gps: GpsSample, frame_index: int) -> GpsSample:
        if not self.enabled or self.start_frame is None or self.end_frame is None:
            return gps
        if frame_index < self.start_frame or frame_index > self.end_frame:
            return gps

        degraded_until = self.start_frame + max(self.degraded_frames, 0)
        if frame_index < degraded_until:
            return replace(
                gps,
                valid=True,
                hdop=max(float(gps.hdop or 0.0), self.degraded_hdop),
                satellites=min(int(gps.satellites or self.degraded_satellites), self.degraded_satellites),
            )
        return replace(gps, valid=False, hdop=None, satellites=0)


class GpsHandoverManager:
    """Tracks GPS fallback transitions, latch state, and re-lock smoothing."""

    def __init__(
        self,
        base_gps_noise_m: float = 3.0,
        relock_noise_multiplier: float = 4.0,
        relock_smoothing_frames: int = 15,
    ) -> None:
        self.base_gps_noise_m = base_gps_noise_m
        self.relock_noise_multiplier = relock_noise_multiplier
        self.relock_smoothing_frames = max(relock_smoothing_frames, 0)
        self.previous_state: Optional[GpsState] = None
        self.last_good_gps: Optional[GpsSample] = None
        self.last_good_xy: Optional[Tuple[float, float]] = None
        self.last_good_theta: Optional[float] = None
        self.latched_gps: Optional[GpsSample] = None
        self.latched_xy: Optional[Tuple[float, float]] = None
        self.latched_theta: Optional[float] = None
        self.lost_frames = 0
        self.relock_frames = 0
        self._relock_remaining = 0

    def update(
        self,
        gps: GpsSample,
        gps_xy: Optional[Tuple[float, float]],
        pose_before_update: Pose2D,
        gps_state: GpsState,
    ) -> HandoverEstimate:
        transition = None
        if self.previous_state is not None and self.previous_state != gps_state:
            transition = f"{self.previous_state}->{gps_state}"

        if transition in {"good->degraded", "degraded->lost", "good->lost"}:
            self._latch_last_good()

        if transition in {"lost->good", "degraded->good"}:
            self._relock_remaining = self.relock_smoothing_frames

        gps_noise_m = self._gps_noise_for_state(gps_state)
        loss_error_m = _pose_distance(pose_before_update, gps_xy) if gps_state == "lost" else None
        relock_error_m = (
            _pose_distance(pose_before_update, gps_xy)
            if transition in {"lost->good", "degraded->good"}
            else None
        )

        if gps_state == "lost":
            self.lost_frames += 1
        else:
            self.lost_frames = 0

        if self._relock_remaining > 0 and gps_state == "good":
            self.relock_frames += 1
            self._relock_remaining -= 1
        elif gps_state != "good":
            self.relock_frames = 0

        if gps_state == "good" and gps.valid and gps_xy is not None:
            self.last_good_gps = gps
            self.last_good_xy = gps_xy
            self.last_good_theta = pose_before_update.theta

        self.previous_state = gps_state
        return HandoverEstimate(
            mode=_mode_for_state(gps_state),
            transition=transition,
            lost_frames=self.lost_frames,
            relock_frames=self.relock_frames,
            gps_correction_noise_m=gps_noise_m,
            latched_gps=self.latched_gps,
            latched_xy=self.latched_xy,
            latched_theta=self.latched_theta,
            loss_error_m=loss_error_m,
            relock_error_m=relock_error_m,
        )

    def _latch_last_good(self) -> None:
        if self.last_good_gps is None or self.last_good_xy is None:
            return
        self.latched_gps = self.last_good_gps
        self.latched_xy = self.last_good_xy
        self.latched_theta = self.last_good_theta

    def _gps_noise_for_state(self, gps_state: GpsState) -> float:
        if gps_state != "good":
            return self.base_gps_noise_m
        if self._relock_remaining <= 0 or self.relock_smoothing_frames <= 0:
            return self.base_gps_noise_m
        ratio = self._relock_remaining / max(self.relock_smoothing_frames, 1)
        return self.base_gps_noise_m * (1.0 + self.relock_noise_multiplier * ratio)


def _mode_for_state(gps_state: GpsState) -> str:
    if gps_state == "good":
        return "gps_fused"
    if gps_state == "degraded":
        return "gps_degraded_vo_ready"
    return "visual_fallback"


def _pose_distance(pose: Pose2D, point: Optional[Tuple[float, float]]) -> Optional[float]:
    if point is None:
        return None
    return float(math.hypot(pose.x - point[0], pose.y - point[1]))
