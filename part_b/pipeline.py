import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional, Tuple

from .detector import Phase4LandmarkDetector
from .dead_reckoning import WheelImuDeadReckoner, choose_motion_delta, integrate_body_delta
from .events import UTurnDetector
from .fusion import LocalizationEKF
from .ghost_projection import project_landmark, reprojection_error_px
from .gps import GpsIntegrityMonitor
from .kitti import KittiRawSequence, discover_kitti_sequences
from .lane import create_lane_detector
from .landmark_db import LandmarkDatabase
from .ocr import create_ocr_backend
from .schema import DeltaPose, LaneEstimate, MotionSource, Phase3Output, Pose2D, ground_delta_from_points
from .vo import OrbVisualOdometry
from .handover import GpsHandoverManager, GpsLossSimulator


def run_sequence(
    sync_path: str,
    calib_path: Optional[str],
    output: str,
    camera: str = "image_02",
    max_frames: Optional[int] = None,
    lane_backend: str = "heuristic",
    lane_model: str = "models/ufldv2_culane_res34.onnx",
    lane_dataset: str = "culane",
    lane_side_mode: str = "binary_road",
    lane_every_n: int = 3,
    landmark_db_path: Optional[str] = None,
    landmark_every_n: int = 5,
    landmark_ocr_lang: str = "vi",
    landmark_reprojection_gate_px: float = 160.0,
    gps_loss_start: Optional[int] = None,
    gps_loss_end: Optional[int] = None,
    gps_loss_degraded_frames: int = 5,
    motion_source: MotionSource = "vo",
) -> None:
    seq = KittiRawSequence(sync_path=sync_path, calib_path=calib_path, camera=camera)
    gps_monitor = GpsIntegrityMonitor()
    lane_detector = create_lane_detector(
        backend=lane_backend,
        model_path=lane_model,
        dataset=lane_dataset,
        lane_side_mode=lane_side_mode,
    )
    vo = OrbVisualOdometry(seq.camera_params)
    dead_reckoner = WheelImuDeadReckoner()
    ekf = LocalizationEKF()
    events = UTurnDetector()
    gps_loss = GpsLossSimulator(
        start_frame=gps_loss_start,
        end_frame=gps_loss_end,
        degraded_frames=gps_loss_degraded_frames,
    )
    handover_manager = GpsHandoverManager()
    landmark_db = None
    landmark_detector = None
    if landmark_db_path:
        landmark_db = LandmarkDatabase.from_jsonl(seq.sequence_id, landmark_db_path)
        landmark_detector = Phase4LandmarkDetector(
            camera_fx=seq.camera_params.fx,
            camera_cx=seq.camera_params.cx,
            ocr=create_ocr_backend(lang=landmark_ocr_lang),
        )

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prev_scale_xy: Optional[Tuple[float, float]] = None
    prev_scale_frame: Optional[int] = None
    prev_heading_xy: Optional[Tuple[float, float]] = None
    last_good_scale_hint = 0.0
    vo_scale_source_counts = {
        "gps_good": 0,
        "last_good_gps_scale": 0,
        "unit_fallback": 0,
    }
    motion_delta_source_counts = {}
    count = 0
    gps_state_counts = {"good": 0, "degraded": 0, "lost": 0}
    lane_counts = {"left": 0, "right": 0}
    valid_vo = 0
    valid_motion = 0
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
    motion_pose = Pose2D(x=0.0, y=0.0, theta=0.0)
    lane_every_n = max(1, int(lane_every_n))
    landmark_every_n = max(1, int(landmark_every_n))
    last_lane: Optional[LaneEstimate] = None
    landmark_observations = 0
    landmark_matches = 0
    landmark_corrections = 0
    landmark_projected = 0
    landmark_reprojection_rejected = 0
    landmark_reprojection_errors = []
    try:
        with out_path.open("w", encoding="utf-8") as f:
            for sample in seq.iter_samples(max_frames=max_frames):
                frame_t0 = time.perf_counter()
                gps_sample = gps_loss.apply(sample.meta.gps, sample.meta.frame_index)
                sample_meta = replace(sample.meta, gps=gps_sample)
                gps_state = gps_monitor.update(gps_sample)
                gps_heading_rad, prev_heading_xy = _gps_course_heading(
                    gps_state=gps_state,
                    curr_xy=sample.local_xy,
                    prev_good_xy=prev_heading_xy,
                )
                wheel_imu_delta = dead_reckoner.update(
                    timestamp=sample_meta.timestamp,
                    speed_mps=sample_meta.gps.speed_mps,
                    gyro_z_rad_s=sample_meta.imu.gz,
                )
                if motion_source == "vo":
                    (
                        scale_hint,
                        scale_source,
                        prev_scale_xy,
                        prev_scale_frame,
                        last_good_scale_hint,
                    ) = _vo_scale_hint(
                        gps_state=gps_state,
                        curr_xy=sample.local_xy,
                        curr_frame=sample.meta.frame_index,
                        prev_good_xy=prev_scale_xy,
                        prev_good_frame=prev_scale_frame,
                        last_good_scale_hint=last_good_scale_hint,
                    )
                else:
                    scale_hint, scale_source = _wheel_imu_scale_hint(wheel_imu_delta)

                t0 = time.perf_counter()
                vo_pose, vo_delta = vo.update(sample.frame, scale_hint=scale_hint)
                vo_ms.append((time.perf_counter() - t0) * 1000.0)
                raw_vo_pose = vo_pose
                if gps_state == "good" and sample.local_xy is not None:
                    vo_pose = Pose2D(
                        x=float(sample.local_xy[0]),
                        y=float(sample.local_xy[1]),
                        theta=float(gps_heading_rad if gps_heading_rad is not None else vo_pose.theta),
                    )
                    vo.pose = vo_pose
                motion_delta, motion_delta_source = choose_motion_delta(
                    motion_source=motion_source,
                    vo_delta=vo_delta,
                    wheel_imu_delta=wheel_imu_delta,
                )
                if motion_source == "vo":
                    pose = vo_pose
                else:
                    motion_pose = integrate_body_delta(motion_pose, motion_delta)
                    pose = motion_pose

                t0 = time.perf_counter()
                if last_lane is None or count % lane_every_n == 0:
                    lane = lane_detector.estimate(sample.frame)
                    last_lane = lane
                    lane_ms.append((time.perf_counter() - t0) * 1000.0)
                else:
                    lane = last_lane
                    lane_ms.append(0.0)

                t0 = time.perf_counter()
                handover = handover_manager.update(
                    gps=gps_sample,
                    gps_xy=sample.local_xy,
                    pose_before_update=prev_fused_pose,
                    gps_state=gps_state,
                )
                gps_xy_for_correction = sample.local_xy if gps_sample.valid else None
                fused_pose = ekf.update(
                    motion_delta,
                    gps_xy_for_correction,
                    gps_state,
                    gps_noise_m=handover.gps_correction_noise_m,
                    gps_heading_rad=gps_heading_rad,
                )
                if landmark_db is not None and landmark_detector is not None and count % landmark_every_n == 0:
                    detected_landmarks = landmark_detector.detect(
                        frame=sample.frame,
                        pose=fused_pose,
                        sequence_id=seq.sequence_id,
                        frame_index=sample.meta.frame_index,
                        timestamp=sample.meta.timestamp,
                    )
                    landmark_observations += len(detected_landmarks)
                    for obs in detected_landmarks:
                        match = landmark_db.find_best_match(obs)
                        if match is not None:
                            landmark_matches += 1
                            record = landmark_db.records[match.landmark_id]
                            frame_h, frame_w = sample.frame.shape[:2]
                            projection = project_landmark(
                                record=record,
                                pose=fused_pose,
                                camera_fx=seq.camera_params.fx,
                                camera_cx=seq.camera_params.cx,
                                image_width=frame_w,
                                image_height=frame_h,
                            )
                            if projection is not None:
                                landmark_projected += 1
                                error_px = reprojection_error_px(projection, obs.bbox_xyxy)
                                landmark_reprojection_errors.append(error_px)
                                if landmark_reprojection_gate_px > 0 and error_px > landmark_reprojection_gate_px:
                                    landmark_reprojection_rejected += 1
                                    landmark_db.upsert(obs)
                                    continue
                            fused_pose = ekf.correct_landmark(obs.p_3D, record.p_3D, match_score=match.score)
                            landmark_corrections += 1
                        landmark_db.upsert(obs)
                event_estimate = events.update(sample.meta.timestamp, fused_pose)
                fusion_ms.append((time.perf_counter() - t0) * 1000.0)

                row = Phase3Output(
                    sample=sample_meta,
                    gps_local_xy=sample.local_xy,
                    pose_local=pose,
                    fused_pose=fused_pose,
                    delta_pose=motion_delta,
                    lane=lane,
                    gps_state=gps_state,
                    events=event_estimate,
                    handover=handover,
                    vo_scale_source=scale_source,
                    vo_scale_hint_m=scale_hint,
                    motion_source=motion_source,
                    motion_delta_source=motion_delta_source,
                    vo_pose_local=raw_vo_pose,
                    vo_delta_pose=vo_delta,
                    wheel_imu_delta_pose=wheel_imu_delta,
                )
                f.write(json.dumps(row.to_jsonable(), ensure_ascii=False) + "\n")
                prev_fused_pose = fused_pose
                count += 1
                vo_scale_source_counts[scale_source] = vo_scale_source_counts.get(scale_source, 0) + 1
                motion_delta_source_counts[motion_delta_source] = (
                    motion_delta_source_counts.get(motion_delta_source, 0) + 1
                )
                gps_state_counts[gps_state] += 1
                if handover.transition:
                    handover_transitions[handover.transition] = handover_transitions.get(handover.transition, 0) + 1
                if handover.loss_error_m is not None:
                    max_loss_error = max(max_loss_error, handover.loss_error_m)
                if handover.relock_error_m is not None:
                    relock_errors.append(handover.relock_error_m)
                lane_counts[lane.lane_side] = lane_counts.get(lane.lane_side, 0) + 1
                if motion_delta.valid:
                    valid_motion += 1
                if vo_delta.valid:
                    valid_vo += 1
                    total_inliers += vo_delta.inliers
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
        "lane_side_mode": lane_side_mode,
        "lane_every_n": lane_every_n,
        "landmarks": {
            "enabled": landmark_db_path is not None,
            "db_path": landmark_db_path,
            "every_n": landmark_every_n,
            "observations": landmark_observations,
            "matches": landmark_matches,
            "corrections": landmark_corrections,
            "ghost_projection": {
                "projected": landmark_projected,
                "rejected_by_gate": landmark_reprojection_rejected,
                "gate_px": landmark_reprojection_gate_px,
                "avg_reprojection_error_px": _mean(landmark_reprojection_errors),
                "p95_reprojection_error_px": _percentile(landmark_reprojection_errors, 0.95),
            },
        },
        "gps_state_counts": gps_state_counts,
        "gps_loss_simulation": {
            "enabled": gps_loss.enabled,
            "start_frame": gps_loss_start,
            "end_frame": gps_loss_end,
            "degraded_frames": gps_loss_degraded_frames,
        },
        "motion": {
            "source": motion_source,
            "delta_source_counts": motion_delta_source_counts,
            "valid_motion_frames": valid_motion,
            "wheel_imu_note": (
                "KITTI/demo speed_mps is the OXTS velocity proxy for wheel CAN speed; "
                "imu.gz is used as yaw-rate. GPS XY is not used for wheel/IMU scale."
            ),
        },
        "vo_scale": {
            "policy": _scale_policy(motion_source),
            "source_counts": vo_scale_source_counts,
            "last_good_scale_hint_m": last_good_scale_hint,
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


def _vo_scale_hint(
    gps_state: str,
    curr_xy: Optional[Tuple[float, float]],
    curr_frame: int,
    prev_good_xy: Optional[Tuple[float, float]],
    prev_good_frame: Optional[int],
    last_good_scale_hint: float,
) -> Tuple[float, str, Optional[Tuple[float, float]], Optional[int], float]:
    """Return an honest monocular VO scale hint for GPS-loss experiments.

    KITTI still has hidden ground-truth GPS local XY even when we simulate GPS
    loss. Only using that delta while GPS is good prevents the handover artifact
    from quietly benefiting from the answer key during degraded/lost frames.
    """

    if gps_state == "good" and curr_xy is not None:
        frame_gap = None if prev_good_frame is None else int(curr_frame) - int(prev_good_frame)
        if prev_good_xy is not None and frame_gap is not None and frame_gap <= 1:
            scale_hint = ground_delta_from_points(prev_good_xy, curr_xy)
            if scale_hint > 0.01:
                last_good_scale_hint = scale_hint
            return scale_hint, "gps_good", curr_xy, int(curr_frame), last_good_scale_hint

        if last_good_scale_hint > 0.01:
            return (
                last_good_scale_hint,
                "last_good_gps_scale",
                curr_xy,
                int(curr_frame),
                last_good_scale_hint,
            )

        scale_hint = 0.0
        if scale_hint > 0.01:
            last_good_scale_hint = scale_hint
        return scale_hint, "gps_good", curr_xy, int(curr_frame), last_good_scale_hint

    if last_good_scale_hint > 0.01:
        return (
            last_good_scale_hint,
            "last_good_gps_scale",
            prev_good_xy,
            prev_good_frame,
            last_good_scale_hint,
        )

    return 0.0, "unit_fallback", prev_good_xy, prev_good_frame, last_good_scale_hint


def _gps_course_heading(
    gps_state: str,
    curr_xy: Optional[Tuple[float, float]],
    prev_good_xy: Optional[Tuple[float, float]],
    min_distance_m: float = 0.5,
) -> Tuple[Optional[float], Optional[Tuple[float, float]]]:
    if gps_state != "good" or curr_xy is None:
        return None, prev_good_xy

    if prev_good_xy is None:
        return None, curr_xy

    dx = float(curr_xy[0] - prev_good_xy[0])
    dy = float(curr_xy[1] - prev_good_xy[1])
    if math.hypot(dx, dy) < min_distance_m:
        return None, prev_good_xy

    # Pose theta uses +Y as vehicle-forward when theta=0.
    return float(math.atan2(-dx, dy)), curr_xy


def _wheel_imu_scale_hint(delta: DeltaPose) -> Tuple[float, str]:
    if delta.valid:
        return delta.scale, "wheel_imu_oxts_proxy"
    return 0.0, "wheel_imu_unavailable"


def _scale_policy(motion_source: MotionSource) -> str:
    if motion_source == "vo":
        return (
            "Use GPS local delta only while GPS state is good; degraded/lost frames use "
            "the last good scale hint, then unit fallback."
        )
    return (
        "Use gps.speed_mps as a KITTI/OXTS proxy for wheel speed and imu.gz for yaw-rate; "
        "no hidden GPS XY scale is used by wheel/IMU modes."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 KITTI VO + lane + GPS baseline")
    parser.add_argument("--data-root", default="data", help="Folder containing KITTI zips or extracted dirs")
    parser.add_argument("--sequence", default=None, help="Sequence id fragment, e.g. 0001 or 2011_09_26_drive_0056")
    parser.add_argument("--sync", default=None, help="Explicit KITTI *_sync.zip or extracted *_sync directory")
    parser.add_argument("--calib", default=None, help="Explicit KITTI calib zip/file")
    parser.add_argument("--camera", default="image_02", choices=["image_02", "image_03"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output JSONL path")
    parser.add_argument("--lane-backend", default="heuristic", choices=["ufldv2", "heuristic"])
    parser.add_argument("--lane-model", default="models/ufldv2_culane_res34.onnx", help="UFLDv2 ONNX lane model")
    parser.add_argument("--lane-dataset", default="culane", choices=["culane", "tusimple", "curvelanes"])
    parser.add_argument(
        "--lane-side-mode",
        default="binary_road",
        choices=["binary_road", "ego_offset"],
        help=(
            "binary_road returns only left/right for the current road lane; "
            "ego_offset keeps the old left/center/right/unknown camera-offset semantics."
        ),
    )
    parser.add_argument("--lane-every-n", type=int, default=3, help="Run lane inference once every N processed frames")
    parser.add_argument("--landmark-db", default=None, help="Optional Phase 4 landmark JSONL for EKF landmark correction")
    parser.add_argument("--landmark-every-n", type=int, default=5, help="Run landmark detection/correction once every N frames")
    parser.add_argument("--landmark-ocr-lang", default="vi")
    parser.add_argument(
        "--landmark-reprojection-gate-px",
        type=float,
        default=160.0,
        help="Reject landmark EKF corrections whose ghost projection error exceeds this many pixels; use 0 to disable",
    )
    parser.add_argument("--gps-loss-start", type=int, default=None, help="First frame to simulate degraded/lost GPS")
    parser.add_argument("--gps-loss-end", type=int, default=None, help="Last frame to simulate degraded/lost GPS")
    parser.add_argument("--gps-loss-degraded-frames", type=int, default=5, help="Frames kept degraded before GPS is lost")
    parser.add_argument(
        "--motion-source",
        default="vo",
        choices=["vo", "wheel_imu", "vo_wheel_imu"],
        help="Odometry source for EKF predict: VO baseline, wheel/IMU dead reckoning, or VO with wheel/IMU scale/yaw",
    )
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
        lane_side_mode=args.lane_side_mode,
        lane_every_n=args.lane_every_n,
        landmark_db_path=args.landmark_db,
        landmark_every_n=args.landmark_every_n,
        landmark_ocr_lang=args.landmark_ocr_lang,
        landmark_reprojection_gate_px=args.landmark_reprojection_gate_px,
        gps_loss_start=args.gps_loss_start,
        gps_loss_end=args.gps_loss_end,
        gps_loss_degraded_frames=args.gps_loss_degraded_frames,
        motion_source=args.motion_source,
    )


if __name__ == "__main__":
    main()
