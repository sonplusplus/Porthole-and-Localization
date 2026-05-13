import argparse
import json
from pathlib import Path
from typing import Optional

from .phase3_events import UTurnDetector
from .phase3_fusion import LocalizationEKF
from .phase3_gps import GpsIntegrityMonitor
from .phase3_kitti import KittiRawSequence, discover_kitti_sequences
from .phase3_lane import LaneDetectorBaseline
from .phase3_schema import Phase3Output, ground_delta_from_points
from .phase3_vo import OrbVisualOdometry


def run_sequence(
    sync_path: str,
    calib_path: Optional[str],
    output: str,
    camera: str = "image_02",
    max_frames: Optional[int] = None,
) -> None:
    seq = KittiRawSequence(sync_path=sync_path, calib_path=calib_path, camera=camera)
    gps_monitor = GpsIntegrityMonitor()
    lane_detector = LaneDetectorBaseline()
    vo = OrbVisualOdometry(seq.camera_params)
    ekf = LocalizationEKF()
    events = UTurnDetector()

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prev_xy = None
    count = 0
    gps_state_counts = {"good": 0, "degraded": 0, "lost": 0}
    lane_counts = {"left": 0, "right": 0, "center": 0, "unknown": 0}
    valid_vo = 0
    total_inliers = 0
    u_turn_count = 0
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for sample in seq.iter_samples(max_frames=max_frames):
                scale_hint = ground_delta_from_points(prev_xy, sample.local_xy)
                pose, delta = vo.update(sample.frame, scale_hint=scale_hint)
                lane = lane_detector.estimate(sample.frame)
                gps_state = gps_monitor.update(sample.meta.gps)
                fused_pose = ekf.update(delta, sample.local_xy, gps_state)
                event_estimate = events.update(sample.meta.timestamp, fused_pose)
                row = Phase3Output(
                    sample=sample.meta,
                    gps_local_xy=sample.local_xy,
                    pose_local=pose,
                    fused_pose=fused_pose,
                    delta_pose=delta,
                    lane=lane,
                    gps_state=gps_state,
                    events=event_estimate,
                )
                f.write(json.dumps(row.to_jsonable(), ensure_ascii=False) + "\n")
                prev_xy = sample.local_xy
                count += 1
                gps_state_counts[gps_state] += 1
                lane_counts[lane.lane_side] = lane_counts.get(lane.lane_side, 0) + 1
                if delta.valid:
                    valid_vo += 1
                    total_inliers += delta.inliers
                if event_estimate.u_turn:
                    u_turn_count += 1
    finally:
        seq.close()

    summary = {
        "sequence": seq.sequence_id,
        "frames": count,
        "output": str(out_path),
        "gps_state_counts": gps_state_counts,
        "lane_counts": lane_counts,
        "valid_vo_frames": valid_vo,
        "avg_vo_inliers": (total_inliers / valid_vo) if valid_vo else 0.0,
        "u_turn_frames": u_turn_count,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 KITTI VO + lane + GPS baseline")
    parser.add_argument("--data-root", default="data", help="Folder containing KITTI zips or extracted dirs")
    parser.add_argument("--sequence", default=None, help="Sequence id fragment, e.g. 0001 or 2011_09_26_drive_0056")
    parser.add_argument("--sync", default=None, help="Explicit KITTI *_sync.zip or extracted *_sync directory")
    parser.add_argument("--calib", default=None, help="Explicit KITTI calib zip/file")
    parser.add_argument("--camera", default="image_02", choices=["image_02", "image_03"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output JSONL path")
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
    )


if __name__ == "__main__":
    main()
