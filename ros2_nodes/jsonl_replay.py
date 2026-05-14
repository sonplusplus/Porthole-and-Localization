from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


JsonRow = Dict[str, Any]


class JsonlCursor:
    """Small replay cursor for JSONL artifacts produced by Part A and Part B."""

    def __init__(self, rows: Iterable[JsonRow], loop: bool = False) -> None:
        self.rows = list(rows)
        self.loop = bool(loop)
        self.index = 0

    @classmethod
    def from_path(cls, path: str, loop: bool = False) -> "JsonlCursor":
        return cls(load_jsonl(path), loop=loop)

    def next(self) -> Optional[JsonRow]:
        if not self.rows:
            return None
        if self.index >= len(self.rows):
            if not self.loop:
                return None
            self.index = 0
        row = self.rows[self.index]
        self.index += 1
        return row


class GroupedFrameCursor:
    """Replay detections grouped by frame_index as one publishable payload."""

    def __init__(self, groups: Iterable[Tuple[int, List[JsonRow]]], loop: bool = False) -> None:
        self.groups = list(groups)
        self.loop = bool(loop)
        self.index = 0

    @classmethod
    def from_detection_jsonl(cls, path: str, loop: bool = False) -> "GroupedFrameCursor":
        return cls(group_by_frame(load_jsonl(path)), loop=loop)

    def next(self) -> Optional[JsonRow]:
        if not self.groups:
            return None
        if self.index >= len(self.groups):
            if not self.loop:
                return None
            self.index = 0
        frame_index, detections = self.groups[self.index]
        self.index += 1
        return {
            "frame_index": frame_index,
            "count": len(detections),
            "detections": detections,
        }


def load_jsonl(path: str) -> List[JsonRow]:
    rows: List[JsonRow] = []
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL artifact not found: {jsonl_path}")

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {jsonl_path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object at {jsonl_path}:{line_no}")
            rows.append(row)
    return rows


def group_by_frame(rows: Iterable[JsonRow]) -> List[Tuple[int, List[JsonRow]]]:
    grouped: Dict[int, List[JsonRow]] = {}
    for fallback_index, row in enumerate(rows):
        frame_index = int(row.get("frame_index", fallback_index))
        grouped.setdefault(frame_index, []).append(row)
    return sorted(grouped.items(), key=lambda item: item[0])


def nested_get(row: JsonRow, path: Tuple[str, ...], default: Any = None) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def pose_from_phase3_row(row: JsonRow) -> Tuple[float, float, float]:
    pose = row.get("fused_pose") if isinstance(row.get("fused_pose"), dict) else {}
    return (
        float(pose.get("x", 0.0)),
        float(pose.get("y", 0.0)),
        float(pose.get("theta", 0.0)),
    )


def yaw_to_quaternion(yaw_rad: float) -> Tuple[float, float, float, float]:
    half = 0.5 * float(yaw_rad)
    return 0.0, 0.0, math.sin(half), math.cos(half)
