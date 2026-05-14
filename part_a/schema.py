from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, median
from typing import Any, Dict, List, Optional
SEG_POT_MODEL_PATH = "models/yolov8s_pothole.onnx"
DEFAULT_DEPTH_MODEL_PATH = "models/depth_anything_v2_vits.onnx"
@dataclass
class Phase2BTiming:
    """Per-frame timing for the CPU video pipeline."""
    frame_index: int
    capture_ts: float
    end_to_end_ms: float
    detect_ms: float
    depth_ms: float
    fps: float
    pothole_count: int
    queue_size: int
    memory_mb: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Phase2BSummary:
    """Aggregated benchmark summary for Phase 2B."""

    frames_read: int
    frames_processed: int
    frames_dropped: int
    avg_fps: float
    p50_end_to_end_ms: float
    p95_end_to_end_ms: float
    avg_detect_ms: float
    avg_depth_ms: float
    peak_memory_mb: Optional[float]
    model_path: str
    depth_model_path: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def build_summary(
    timings: List[Phase2BTiming],
    frames_read: int,
    frames_dropped: int,
    model_path: str,
    depth_model_path: str,
) -> Phase2BSummary:
    end_to_end = [t.end_to_end_ms for t in timings]
    detect = [t.detect_ms for t in timings]
    depth = [t.depth_ms for t in timings]
    fps_values = [t.fps for t in timings]
    memory = [t.memory_mb for t in timings if t.memory_mb is not None]

    return Phase2BSummary(
        frames_read=frames_read,
        frames_processed=len(timings),
        frames_dropped=frames_dropped,
        avg_fps=float(mean(fps_values)) if fps_values else 0.0,
        p50_end_to_end_ms=float(median(end_to_end)) if end_to_end else 0.0,
        p95_end_to_end_ms=percentile(end_to_end, 0.95),
        avg_detect_ms=float(mean(detect)) if detect else 0.0,
        avg_depth_ms=float(mean(depth)) if depth else 0.0,
        peak_memory_mb=max(memory) if memory else None,
        model_path=model_path,
        depth_model_path=depth_model_path,
    )
