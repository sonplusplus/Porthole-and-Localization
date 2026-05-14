import argparse
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


Point2D = Tuple[float, float]


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {
            "frames": 0,
            "duration_sec": 0.0,
            "error": "empty phase3 output",
        }

    sequence_id = rows[0].get("sample", {}).get("sequence_id", "unknown")
    timestamps = [_get_float(row, ("sample", "timestamp")) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    duration_sec = max(timestamps) - min(timestamps) if len(timestamps) >= 2 else 0.0

    lane_counts = Counter(_get(row, ("lane", "lane_side"), "unknown") for row in rows)
    gps_state_counts = Counter(_get(row, ("gps_state",), "unknown") for row in rows)
    vo_valid = [bool(_get(row, ("delta_pose", "valid"), False)) for row in rows]
    vo_matches = [_get_float(row, ("delta_pose", "matches")) for row in rows]
    vo_inliers = [_get_float(row, ("delta_pose", "inliers")) for row in rows]
    vo_scale_sources = Counter(_get(row, ("vo_scale_source",), "unknown") for row in rows)
    lane_conf = [_get_float(row, ("lane", "confidence")) for row in rows]
    heading_delta = [_get_float(row, ("events", "heading_delta_deg")) for row in rows]
    handover_modes = Counter(_get(row, ("handover", "mode"), "unknown") for row in rows)
    handover_transitions = Counter(
        _get(row, ("handover", "transition"))
        for row in rows
        if _get(row, ("handover", "transition")) is not None
    )

    gps_points = [_point(row.get("gps_local_xy")) for row in rows]
    vo_points = [_pose_point(row.get("pose_local")) for row in rows]
    fused_points = [_pose_point(row.get("fused_pose")) for row in rows]

    good_gps_indexes = [
        i
        for i, row in enumerate(rows)
        if row.get("gps_state") == "good" and gps_points[i] is not None
    ]
    fused_errors = [_distance(fused_points[i], gps_points[i]) for i in good_gps_indexes]
    vo_errors = [_distance(vo_points[i], gps_points[i]) for i in good_gps_indexes]
    fused_errors = [value for value in fused_errors if value is not None]
    vo_errors = [value for value in vo_errors if value is not None]

    u_turn_frames = [
        int(_get(row, ("sample", "frame_index"), -1))
        for row in rows
        if bool(_get(row, ("events", "u_turn"), False))
    ]
    loss_errors = [
        _distance(fused_points[i], gps_points[i])
        for i, row in enumerate(rows)
        if row.get("gps_state") == "lost" and gps_points[i] is not None
    ]
    loss_errors = [value for value in loss_errors if value is not None]
    relock_errors = [
        _get_float(row, ("handover", "relock_error_m"))
        for row in rows
        if _get_float(row, ("handover", "relock_error_m")) is not None
    ]

    metrics = {
        "sequence": sequence_id,
        "frames": len(rows),
        "frame_start": _get(rows[0], ("sample", "frame_index")),
        "frame_end": _get(rows[-1], ("sample", "frame_index")),
        "duration_sec": duration_sec,
        "source_fps": ((len(rows) - 1) / duration_sec) if duration_sec > 0 else 0.0,
        "gps_state_counts": dict(gps_state_counts),
        "handover_modes": dict(handover_modes),
        "handover_transitions": dict(handover_transitions),
        "visual_fallback_error_vs_hidden_gps_m": _error_summary(loss_errors),
        "relock_error_m": _error_summary(relock_errors),
        "lane_counts": dict(lane_counts),
        "lane_unknown_rate": lane_counts.get("unknown", 0) / len(rows),
        "avg_lane_confidence": _safe_mean(lane_conf),
        "vo_valid_frames": sum(vo_valid),
        "vo_valid_ratio": sum(vo_valid) / len(rows),
        "avg_vo_matches": _safe_mean(vo_matches),
        "avg_vo_inliers": _safe_mean(vo_inliers),
        "p50_vo_inliers": _percentile(vo_inliers, 0.50),
        "p95_vo_inliers": _percentile(vo_inliers, 0.95),
        "vo_scale_source_counts": dict(vo_scale_sources),
        "gps_path_length_m": _path_length(gps_points),
        "vo_path_length_m": _path_length(vo_points),
        "fused_path_length_m": _path_length(fused_points),
        "vo_error_vs_gps_m": _error_summary(vo_errors),
        "fused_error_vs_gps_m": _error_summary(fused_errors),
        "final_vo_error_m": _last_error(vo_points, gps_points),
        "final_fused_error_m": _last_error(fused_points, gps_points),
        "u_turn_frames": u_turn_frames,
        "u_turn_count": len(u_turn_frames),
        "max_heading_delta_deg": max([value for value in heading_delta if value is not None], default=0.0),
    }
    metrics["phase3_status"] = _status_notes(metrics)
    return metrics


def write_plots(rows: Sequence[Dict[str, Any]], metrics: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = metrics.get("sequence") or "phase3"
    paths = {
        "trajectory_plot": str(out_dir / f"{stem}_trajectory.png"),
        "error_plot": str(out_dir / f"{stem}_error.png"),
        "timeline_plot": str(out_dir / f"{stem}_timeline.png"),
    }

    _plot_trajectory(rows, paths["trajectory_plot"])
    _plot_error(rows, paths["error_plot"])
    _plot_timeline(rows, paths["timeline_plot"])
    return paths


def _plot_trajectory(rows: Sequence[Dict[str, Any]], output: str) -> None:
    gps = [_point(row.get("gps_local_xy")) for row in rows]
    vo = [_pose_point(row.get("pose_local")) for row in rows]
    fused = [_pose_point(row.get("fused_pose")) for row in rows]

    plt.figure(figsize=(8, 6))
    _plot_xy(gps, "GPS local", "#2f6fdd", linewidth=2.0)
    _plot_xy(vo, "VO", "#d66a2f", linewidth=1.5)
    _plot_xy(fused, "EKF fused", "#2f9e44", linewidth=1.8)
    plt.title("Phase 3 trajectory")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _plot_error(rows: Sequence[Dict[str, Any]], output: str) -> None:
    frames: List[int] = []
    vo_errors: List[float] = []
    fused_errors: List[float] = []
    for row in rows:
        gps = _point(row.get("gps_local_xy"))
        if gps is None:
            continue
        frame = int(_get(row, ("sample", "frame_index"), len(frames)))
        vo_error = _distance(_pose_point(row.get("pose_local")), gps)
        fused_error = _distance(_pose_point(row.get("fused_pose")), gps)
        if vo_error is None or fused_error is None:
            continue
        frames.append(frame)
        vo_errors.append(vo_error)
        fused_errors.append(fused_error)

    plt.figure(figsize=(9, 4.8))
    plt.plot(frames, vo_errors, label="VO vs GPS", color="#d66a2f", linewidth=1.5)
    plt.plot(frames, fused_errors, label="EKF fused vs GPS", color="#2f9e44", linewidth=1.8)
    plt.title("Phase 3 localization error")
    plt.xlabel("frame")
    plt.ylabel("error (m)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _plot_timeline(rows: Sequence[Dict[str, Any]], output: str) -> None:
    frame_indexes = [int(_get(row, ("sample", "frame_index"), index)) for index, row in enumerate(rows)]
    gps_map = {"good": 2, "degraded": 1, "lost": 0}
    lane_map = {"left": -1, "center": 0, "right": 1, "unknown": -2}
    gps_values = [gps_map.get(str(row.get("gps_state")), -1) for row in rows]
    lane_values = [lane_map.get(str(_get(row, ("lane", "lane_side"), "unknown")), -2) for row in rows]
    u_turn_values = [1 if bool(_get(row, ("events", "u_turn"), False)) else 0 for row in rows]

    fig, axes = plt.subplots(3, 1, figsize=(9, 6), sharex=True)
    axes[0].step(frame_indexes, gps_values, where="post", color="#2f6fdd")
    axes[0].set_yticks([0, 1, 2])
    axes[0].set_yticklabels(["lost", "degraded", "good"])
    axes[0].set_ylabel("GPS")
    axes[0].grid(True, alpha=0.25)

    axes[1].step(frame_indexes, lane_values, where="post", color="#6f42c1")
    axes[1].set_yticks([-2, -1, 0, 1])
    axes[1].set_yticklabels(["unknown", "left", "center", "right"])
    axes[1].set_ylabel("lane")
    axes[1].grid(True, alpha=0.25)

    axes[2].step(frame_indexes, u_turn_values, where="post", color="#d6336c")
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(["no", "yes"])
    axes[2].set_ylabel("U-turn")
    axes[2].set_xlabel("frame")
    axes[2].grid(True, alpha=0.25)

    fig.suptitle("Phase 3 state timeline")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)


def _plot_xy(points: Sequence[Optional[Point2D]], label: str, color: str, linewidth: float) -> None:
    clean = [point for point in points if point is not None]
    if not clean:
        return
    xs = [point[0] for point in clean]
    ys = [point[1] for point in clean]
    plt.plot(xs, ys, label=label, color=color, linewidth=linewidth)
    plt.scatter([xs[0]], [ys[0]], color=color, s=24)
    plt.scatter([xs[-1]], [ys[-1]], color=color, marker="x", s=32)


def _status_notes(metrics: Dict[str, Any]) -> List[str]:
    notes: List[str] = []
    if metrics.get("vo_valid_ratio", 0.0) < 0.90:
        notes.append("VO valid ratio is below 90%; inspect feature tracking failures.")
    if metrics.get("lane_unknown_rate", 0.0) > 0.25:
        notes.append("Lane detector returns unknown often; this is expected for a heuristic baseline.")
    fused_p95 = metrics.get("fused_error_vs_gps_m", {}).get("p95")
    if fused_p95 is not None and fused_p95 > 10.0:
        notes.append("Fused pose p95 error is high for GPS-good frames; tune EKF noise or VO scale.")
    if metrics.get("gps_state_counts", {}).get("lost", 0) == 0:
        notes.append("No GPS-lost frames found; run Phase 5 with --gps-loss-start/--gps-loss-end to test handover.")
    relock_count = metrics.get("relock_error_m", {}).get("count", 0)
    if metrics.get("gps_state_counts", {}).get("lost", 0) > 0 and relock_count == 0:
        notes.append("GPS loss was simulated but no re-lock transition was observed; extend frames past --gps-loss-end.")
    if not notes:
        if metrics.get("gps_state_counts", {}).get("lost", 0) > 0:
            notes.append("Phase 5 GPS-loss handover produced fallback and re-lock metrics.")
        else:
            notes.append("Phase 3 baseline has quantified output and is ready for Phase 5 GPS-loss simulation.")
    return notes


def _error_summary(values: Sequence[float]) -> Dict[str, Optional[float]]:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    if not clean:
        return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
    return {
        "count": len(clean),
        "mean": float(mean(clean)),
        "median": float(median(clean)),
        "p95": _percentile(clean, 0.95),
        "max": max(clean),
    }


def _path_length(points: Sequence[Optional[Point2D]]) -> float:
    total = 0.0
    prev: Optional[Point2D] = None
    for point in points:
        if point is None:
            continue
        if prev is not None:
            total += math.hypot(point[0] - prev[0], point[1] - prev[1])
        prev = point
    return float(total)


def _last_error(
    estimated: Sequence[Optional[Point2D]],
    reference: Sequence[Optional[Point2D]],
) -> Optional[float]:
    for est, ref in zip(reversed(estimated), reversed(reference)):
        value = _distance(est, ref)
        if value is not None:
            return value
    return None


def _distance(a: Optional[Point2D], b: Optional[Point2D]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def _point(value: Any) -> Optional[Point2D]:
    if value is None or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def _pose_point(value: Any) -> Optional[Point2D]:
    if not isinstance(value, dict):
        return None
    return float(value.get("x", 0.0)), float(value.get("y", 0.0))


def _safe_mean(values: Iterable[Optional[float]]) -> float:
    clean = [float(value) for value in values if value is not None and math.isfinite(value)]
    return float(mean(clean)) if clean else 0.0


def _percentile(values: Iterable[Optional[float]], q: float) -> Optional[float]:
    clean = sorted(float(value) for value in values if value is not None and math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return float(clean[lower] * (1.0 - weight) + clean[upper] * weight)


def _get(row: Dict[str, Any], path: Tuple[str, ...], default: Any = None) -> Any:
    value: Any = row
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def _get_float(row: Dict[str, Any], path: Tuple[str, ...]) -> Optional[float]:
    value = _get(row, path)
    if value is None:
        return None
    return float(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Phase 3 metrics and plots from JSONL output")
    parser.add_argument("--input", required=True, help="Phase 3 JSONL output")
    parser.add_argument("--metrics", default=None, help="Output metrics JSON path")
    parser.add_argument("--plot-dir", default=None, help="Optional output folder for PNG plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    metrics = compute_metrics(rows)
    if args.plot_dir:
        metrics["plots"] = write_plots(rows, metrics, args.plot_dir)

    metrics_path = Path(args.metrics) if args.metrics else Path(args.input).with_suffix(".metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
