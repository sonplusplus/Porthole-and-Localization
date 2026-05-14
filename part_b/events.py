import math
from collections import deque
from typing import Deque, Tuple

from .schema import EventEstimate, Pose2D


class UTurnDetector:
    def __init__(self, window_sec: float = 2.0, threshold_deg: float = 150.0) -> None:
        self.window_sec = window_sec
        self.threshold_rad = math.radians(threshold_deg)
        self.history: Deque[Tuple[float, float]] = deque()

    def update(self, timestamp: float, pose: Pose2D) -> EventEstimate:
        self.history.append((timestamp, pose.theta))
        while self.history and timestamp - self.history[0][0] > self.window_sec:
            self.history.popleft()

        if len(self.history) < 2:
            return EventEstimate(u_turn=False, heading_delta_deg=0.0)

        oldest_theta = self.history[0][1]
        delta = abs(_wrap_angle(pose.theta - oldest_theta))
        return EventEstimate(
            u_turn=delta >= self.threshold_rad,
            heading_delta_deg=math.degrees(delta),
        )


def _wrap_angle(value: float) -> float:
    return float((value + math.pi) % (2 * math.pi) - math.pi)
