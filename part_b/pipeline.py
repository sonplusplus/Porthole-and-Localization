import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from .events import UTurnDetector
from .fusion import LocalizationEKF
from .gps import GpsIntegrityMonitor
from .kitti import KittiRawSequence, discover_kitti_sequences
from .lane import create_lane_detector
from .schema import Phase3Output, Pose2D, ground_delta_from_points
from .vo import OrbVisualOdometry
from .handover import GpsHandoverManager, GpsLossSimulator


def run_sequence(
    sync_path: str,
    calib_path: Optional[str],
    output: str,
    camera: str = "image_02",
    max_frames: Optional[int] = None,
    lane_backend: str = "ufldv2",
    lane_model: str = "models/ufldv2_culane_res34.onnx",
    lane_dataset: str = "culane",
    gps_loss_start: Optional[int] = None,
    gps_loss_end: Optional[int] = None,
    gps_loss_degraded_frames: int = 5,
) -> None:
    seq = KittiRawSequence(sync_path=sync_path, calib_path=calib_path, camera=camera)
    gps_monitor = GpsIntegrityMonitor()
    lane_detector = create_lane_detector(backend=lane_backend, model_path=lane_model, dataset=lane_dataset)
    vo = OrbVisualOdometry(seq.camera_params)
    ekf = LocalizationEKF()
    events = UTurnDetector()
    gps_loss = GpsLossSimulator(
        start_frame=gps_loss_start,
        end_frame=gps_loss_end,
        degraded_frames=gps_loss_degraded_frames,
    )
    handover_manager = GpsHandoverManager()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prev_xy = None
    count = 0
    gps_state_counts = {"good": 0, "degraded": 0, "lost": 0}
    lane_counts = {"left": 0, "right": 0, "center": 0, "unknown": 0}
    valid_vo = 0
    total_inliers = 0
    u_turn_count = 0
    total_ms = []
    vo_ms = []
    lane_ms = []
    fusion_ms = []
    handover_transitions = {}
    relock_errors = []
    max_loss_error = 0.0
    prev_fused_pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for sample in seq.iter_samples(max_frames=max_frames):
                frame_t0 = time.perf_counter()
                scale_hint = ground_delta_from_points(prev_xy, sample.local_xy)
                gps_sample = gps_loss.apply(sample.meta.gps, sample.meta.frame_index)
                sample_meta = replace(sample.meta, gps=gps_sample)

                t0 = time.perf_counter()
                pose, delta = vo.update(sample.frame, scale_hint=scale_hint)
                vo_ms.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                lane = lane_detector.estimate(sample.frame)
                lane_ms.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                gps_state = gps_monitor.update(gps_sample)
                handover = handover_manager.update(
                    gps=gps_sample,
                    gps_xy=sample.local_xy,
                    pose_before_update=prev_fused_pose,
                    gps_state=gps_state,
                )
                gps_xy_for_correction = sample.local_xy if gps_sample.valid else None
                fused_pose = ekf.update(
                    delta,
                    gps_xy_for_correction,
                    gps_state,
                    gps_noise_m=handover.gps_correction_noise_m,
                )
                event_estimate = events.update(sample.meta.timestamp, fused_pose)
                fusion_ms.append((time.perf_counter() - t0) * 1000.0)

                row = Phase3Output(
                    sample=sample_meta,
                    gps_local_xy=sample.local_xy,
                    pose_local=pose,
                    fused_pose=fused_pose,
                    delta_pose=delta,
                    lane=lane,
                    gps_state=gps_state,
                    events=event_estimate,
                    handover=handover,
                )
                f.write(json.dumps(row.to_jsonable(), ensure_ascii=False) + "\n")
                prev_xy = sample.local_xy
                prev_fused_pose = fused_pose
                count += 1
                gps_state_counts[gps_state] += 1
                if handover.transition:
                    handover_transitions[handover.transition] = handover_transitions.get(handover.transition, 0) + 1
                if handover.loss_error_m is not None:
                    max_loss_error = max(max_loss_error, handover.loss_error_m)
                if handover.relock_error_m is not None:
                    relock_errors.append(handover.relock_error_m)
                lane_counts[lane.lane_side] = lane_counts.get(lane.lane_side, 0) + 1
                if delta.valid:
                    valid_vo += 1
                    total_inliers += delta.inliers
                if event_estimate.u_turn:
                    u_turn_count += 1
                total_ms.append((time.perf_counter() - frame_t0) * 1000.0)
    finally:
        seq.close()

    avg_total_ms = _mean(total_ms)
    summary = {
        "sequence": seq.sequence_id,
        "frames": count,
        "output": str(out_path),
        "lane_backend": lane_backend,
        "lane_model": lane_model,
        "lane_dataset": lane_dataset,
        "gps_state_counts": gps_state_counts,
        "gps_loss_simulation": {
            "enabled": gps_loss.enabled,
            "start_frame": gps_loss_start,
            "end_frame": gps_loss_end,
            "degraded_frames": gps_loss_degraded_frames,
        },
        "handover_transitions": handover_transitions,
        "max_visual_fallback_error_m": max_loss_error,
        "relock_error_m": {
            "count": len(relock_errors),
            "max": max(relock_errors) if relock_errors else None,
            "avg": _mean(relock_errors),
        },
        "lane_counts": lane_counts,
        "valid_vo_frames": valid_vo,
        "avg_vo_inliers": (total_inliers / valid_vo) if valid_vo else 0.0,
        "u_turn_frames": u_turn_count,
        "processing_timing_ms": {
            "avg_total": avg_total_ms,
            "p95_total": _percentile(total_ms, 0.95),
            "avg_vo": _mean(vo_ms),
            "p95_vo": _percentile(vo_ms, 0.95),
            "avg_lane": _mean(lane_ms),
            "p95_lane": _percentile(lane_ms, 0.95),
            "avg_fusion_event": _mean(fusion_ms),
            "p95_fusion_event": _percentile(fusion_ms, 0.95),
        },
        "estimated_processing_fps": (1000.0 / avg_total_ms) if avg_total_ms > 0 else 0.0,
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"Phase 3 baseline complete: sequence={seq.sequence_id}, "
        f"frames={count}, output={out_path}, summary={summary_path}"
    )


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


def _mean(values):
    return float(sum(values) / len(values)) if values else 0.0


def _percentile(values, q: float):
    if not values:
        return 0.0
    clean = sorted(float(value) for value in values)
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(clean) - 1)
    weight = position - lower
    return float(clean[lower] * (1.0 - weight) + clean[upper] * weight)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 KITTI VO + lane + GPS baseline")
    parser.add_argument("--data-root", default="data", help="Folder containing KITTI zips or extracted dirs")
    parser.add_argument("--sequence", default=None, help="Sequence id fragment, e.g. 0001 or 2011_09_26_drive_0056")
    parser.add_argument("--sync", default=None, help="Explicit KITTI *_sync.zip or extracted *_sync directory")
    parser.add_argument("--calib", default=None, help="Explicit KITTI calib zip/file")
    parser.add_argument("--camera", default="image_02", choices=["image_02", "image_03"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--lane-backend", default="ufldv2", choices=["ufldv2", "heuristic"])
    parser.add_argument("--lane-model", default="models/ufldv2_culane_res34.onnx", help="UFLDv2 ONNX lane model")
    parser.add_argument("--lane-dataset", default="culane", choices=["culane", "tusimple", "curvelanes"])
    parser.add_argument("--gps-loss-start", type=int, default=None, help="First frame to simulate degraded/lost GPS")
    parser.add_argument("--gps-loss-end", type=int, default=None, help="Last frame to simulate degraded/lost GPS")
    parser.add_argument("--gps-loss-degraded-frames", type=int, default=5, help="Frames kept degraded before GPS is lost")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sync_path, calib_path = resolve_sequence(args)
    seq_id = Path(sync_path).name.replace("_sync.zip", "").replace("_sync", "")
    output = args.output or str(Path("data") / "phase3_outputs" / f"{seq_id}_{args.camera}.jsonl")
    run_sequence(
        sync_path=sync_path,
        calib_path=calib_path,
        output=output,
        camera=args.camera,
        max_frames=args.max_frames,
        lane_backend=args.lane_backend,
        lane_model=args.lane_model,
        lane_dataset=args.lane_dataset,
        gps_loss_start=args.gps_loss_start,
        gps_loss_end=args.gps_loss_end,
        gps_loss_degraded_frames=args.gps_loss_degraded_frames,
    )


if __name__ == "__main__":
    main()
