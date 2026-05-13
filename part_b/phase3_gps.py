from .phase3_schema import GpsSample, GpsState


class GpsIntegrityMonitor:
    def __init__(
        self,
        hdop_degraded: float = 5.0,
        min_satellites: int = 4,
        lost_after_invalid: int = 2,
    ) -> None:
        self.hdop_degraded = hdop_degraded
        self.min_satellites = min_satellites
        self.lost_after_invalid = lost_after_invalid
        self.invalid_count = 0

    def update(self, gps: GpsSample) -> GpsState:
        if not gps.valid:
            self.invalid_count += 1
            return "lost" if self.invalid_count >= self.lost_after_invalid else "degraded"

        self.invalid_count = 0
        if gps.hdop is not None and gps.hdop > self.hdop_degraded:
            return "degraded"
        if gps.satellites is not None and gps.satellites < self.min_satellites:
            return "degraded"
        return "good"
