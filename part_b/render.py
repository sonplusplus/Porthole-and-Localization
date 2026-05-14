import argparse
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .kitti import KittiRawSequence, discover_kitti_sequences
from .metrics import load_rows


Color = Tuple[int, int, int]
Point2D = Tuple[float, float]

BLUE: Color = (220, 120, 40)
GREEN: Color = (70, 180, 80)
ORANGE: Color = (40, 130, 230)
WHITE: Color = (245, 245, 245)
BLACK: Color = (20, 20, 20)
MAGENTA: Color = (190, 70, 210)


def render_overlay(
    sync_path: str,
    calib_path: Optional[str],
    phase3_output: str,
    output: str,
    camera: str = "image_02",
    max_frames: Optional[int] = None,
) -> None:
    rows = load_rows(phase3_output)
    rows_by_frame = {
        int(row.get("sample", {}).get("frame_index", index)): row
        for index, row in enumerate(rows)
    }
    trajectory_points = _collect_trajectory(rows)

    seq = KittiRawSequence(sync_path=sync_path, calib_path=calib_path, camera=camera)
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer: Optional[cv2.VideoWriter] = None
    written = 0
    try:
        fps = _estimate_fps(rows)
        for sample in seq.iter_samples(max_frames=max_frames):
            row = rows_by_frame.get(sample.meta.frame_index)
            if row is None:
                continue
            frame = sample.frame.copy()
            _draw_phase3_overlay(frame, row, trajectory_points)

            if writer is None:
                h, w = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
            writer.write(frame)
            written += 1
    finally:
        seq.close()
        if writer is not None:
            writer.release()

    print(f"Phase 3 overlay rendered: frames={written}, output={out_path}")


def _draw_phase3_overlay(
    frame: np.ndarray,
    row: Dict[str, Any],
    trajectory_points: Dict[str, List[Optional[Point2D]]],
) -> None:
    sample = row.get("sample", {})
    lane = row.get("lane", {})
    delta = row.get("delta_pose", {})
    vo_delta = row.get("vo_delta_pose") if isinstance(row.get("vo_delta_pose"), dict) else delta
    fused = row.get("fused_pose", {})
    event = row.get("events", {})

    frame_index = int(sample.get("frame_index", 0))
    sequence = sample.get("sequence_id", "unknown")
    gps_state = str(row.get("gps_state", "unknown"))
    motion_delta_source = str(row.get("motion_delta_source") or row.get("motion_source") or "vo")
    lane_side = str(lane.get("lane_side", "unknown"))
    lane_conf = float(lane.get("confidence") or 0.0)
    motion_valid = bool(delta.get("valid", False))
    vo_valid = bool(vo_delta.get("valid", False))
    matches = int(vo_delta.get("matches") or 0)
    inliers = int(vo_delta.get("inliers") or 0)
    theta_deg = math.degrees(float(fused.get("theta") or 0.0))
    u_turn = bool(event.get("u_turn", False))

    lane_center = lane.get("lane_center_px")
    if lane_center is not None:
        x = int(float(lane_center))
        cv2.line(frame, (x, int(frame.shape[0] * 0.55)), (x, frame.shape[0] - 1), MAGENTA, 2)

    lines = [
        f"Phase 3 KITTI | {sequence} | frame {frame_index}",
        f"GPS: {gps_state} | lane: {lane_side} ({lane_conf:.2f})",
        (
            f"Motion: {motion_delta_source} ({'valid' if motion_valid else 'invalid'}) | "
            f"VO {'valid' if vo_valid else 'invalid'} {matches}/{inliers}"
        ),
        f"Fused pose: x={float(fused.get('x') or 0.0):.1f}m y={float(fused.get('y') or 0.0):.1f}m th={theta_deg:.1f}deg",
        f"U-turn: {'YES' if u_turn else 'no'} | heading delta {float(event.get('heading_delta_deg') or 0.0):.1f}deg",
    ]
    _draw_text_block(frame, lines, origin=(18, 28))
    _draw_minimap(frame, trajectory_points, frame_index)


def _draw_text_block(frame: np.ndarray, lines: Sequence[str], origin: Tuple[int, int]) -> None:
    x, y = origin
    line_h = 24
    width = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 1)[0][0] for line in lines) + 22
    height = line_h * len(lines) + 14
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y - 22), (x - 10 + width, y - 22 + height), BLACK, -1)
    cv2.addWeighted(overlay, 0.68, frame, 0.32, 0, frame)
    for index, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y + index * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            WHITE,
            1,
            cv2.LINE_AA,
        )


def _draw_minimap(
    frame: np.ndarray,
    trajectory_points: Dict[str, List[Optional[Point2D]]],
    frame_index: int,
) -> None:
    map_w, map_h = 260, 210
    margin = 16
    x0 = frame.shape[1] - map_w - margin
    y0 = frame.shape[0] - map_h - margin
    if x0 < 0 or y0 < 0:
        return

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + map_w, y0 + map_h), BLACK, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + map_w, y0 + map_h), (80, 80, 80), 1)
    cv2.putText(frame, "trajectory", (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1, cv2.LINE_AA)

    bounds = _trajectory_bounds(trajectory_points)
    if bounds is None:
        return

    frame_indexes = trajectory_points["frame"]
    upto = 0
    for idx, value in enumerate(frame_indexes):
        if value is not None and value <= frame_index:
            upto = idx + 1

    _draw_path(frame, trajectory_points["gps"][:upto], bounds, (x0, y0, map_w, map_h), BLUE)
    _draw_path(frame, trajectory_points["odom"][:upto], bounds, (x0, y0, map_w, map_h), ORANGE)
    _draw_path(frame, trajectory_points["fused"][:upto], bounds, (x0, y0, map_w, map_h), GREEN)

    _legend(frame, x0 + 10, y0 + map_h - 54, "GPS", BLUE)
    _legend(frame, x0 + 88, y0 + map_h - 54, "Odom", ORANGE)
    _legend(frame, x0 + 152, y0 + map_h - 54, "Fused", GREEN)


def _draw_path(
    frame: np.ndarray,
    points: Sequence[Optional[Point2D]],
    bounds: Tuple[float, float, float, float],
    rect: Tuple[int, int, int, int],
    color: Color,
) -> None:
    pixels = [_map_point(point, bounds, rect) for point in points if point is not None]
    if len(pixels) < 2:
        return
    for a, b in zip(pixels, pixels[1:]):
        cv2.line(frame, a, b, color, 2)
    cv2.circle(frame, pixels[-1], 4, color, -1)


def _legend(frame: np.ndarray, x: int, y: int, label: str, color: Color) -> None:
    cv2.line(frame, (x, y), (x + 18, y), color, 2)
    cv2.putText(frame, label, (x + 22, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA)


def _map_point(
    point: Point2D,
    bounds: Tuple[float, float, float, float],
    rect: Tuple[int, int, int, int],
) -> Tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    x0, y0, w, h = rect
    pad = 30
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    px = x0 + pad + int((point[0] - min_x) / span_x * max(w - 2 * pad, 1))
    py = y0 + h - pad - int((point[1] - min_y) / span_y * max(h - 2 * pad, 1))
    return px, py


def _trajectory_bounds(points: Dict[str, List[Optional[Point2D]]]) -> Optional[Tuple[float, float, float, float]]:
    clean = [point for key in ("gps", "odom", "fused") for point in points[key] if point is not None]
    if not clean:
        return None
    xs = [point[0] for point in clean]
    ys = [point[1] for point in clean]
    return min(xs), max(xs), min(ys), max(ys)


def _collect_trajectory(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Optional[Point2D]]]:
    return {
        "frame": [int(row.get("sample", {}).get("frame_index", index)) for index, row in enumerate(rows)],
        "gps": [_point(row.get("gps_local_xy")) for row in rows],
        "odom": [_pose_point(row.get("pose_local")) for row in rows],
        "fused": [_pose_point(row.get("fused_pose")) for row in rows],
    }


def _estimate_fps(rows: Sequence[Dict[str, Any]]) -> float:
    timestamps = [
        float(row.get("sample", {}).get("timestamp"))
        for row in rows
        if row.get("sample", {}).get("timestamp") is not None
    ]
    if len(timestamps) < 2:
        return 10.0
    duration = max(timestamps) - min(timestamps)
    if duration <= 0:
        return 10.0
    return max(1.0, min(60.0, (len(timestamps) - 1) / duration))


def _point(value: Any) -> Optional[Point2D]:
    if value is None or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def _pose_point(value: Any) -> Optional[Point2D]:
    if not isinstance(value, dict):
        return None
    return float(value.get("x", 0.0)), float(value.get("y", 0.0))


def resolve_sequence(args: argparse.Namespace):
    if args.sync:
        return args.sync, args.calib

    refs = discover_kitti_sequences(args.data_root)
    if not refs:
        raise FileNotFoundError(f"No KITTI *_sync.zip or *_sync directories found under {args.data_root}")

    if args.sequence:
        needle = args.sequence
        for ref in refs:
            if needle in ref.sequence_id:
                return str(ref.sync_path), str(ref.calib_path) if ref.calib_path else None
        raise FileNotFoundError(f"Could not find sequence matching: {args.sequence}")

    ref = refs[0]
    return str(ref.sync_path), str(ref.calib_path) if ref.calib_path else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Phase 3 KITTI overlay video")
    parser.add_argument("--phase3-output", required=True, help="Phase 3 JSONL produced by pipeline")
    parser.add_argument("--data-root", default="data", help="Folder containing KITTI zips or extracted dirs")
    parser.add_argument("--sequence", default=None, help="Sequence id fragment, e.g. 0001 or 0056")
    parser.add_argument("--sync", default=None, help="Explicit KITTI *_sync.zip or extracted *_sync directory")
    parser.add_argument("--calib", default=None, help="Explicit KITTI calib zip/file")
    parser.add_argument("--camera", default="image_02", choices=["image_02", "image_03"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output MP4 path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sync_path, calib_path = resolve_sequence(args)
    seq_id = Path(sync_path).name.replace("_sync.zip", "").replace("_sync", "")
    output = args.output or str(Path("data") / "phase3_outputs" / f"{seq_id}_{args.camera}_overlay.mp4")
    render_overlay(
        sync_path=sync_path,
        calib_path=calib_path,
        phase3_output=args.phase3_output,
        output=output,
        camera=args.camera,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
