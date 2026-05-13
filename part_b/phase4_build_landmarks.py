import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from .phase3_kitti import KittiRawSequence, discover_kitti_sequences
from .phase3_schema import Pose2D
from .phase4_detector import Phase4LandmarkDetector
from .phase4_landmark_db import LandmarkDatabase
from .phase4_ocr import create_ocr_backend


def build_landmark_db(
    sync_path: str,
    calib_path: Optional[str],
    output: str,
    phase3_output: Optional[str] = None,
    camera: str = "image_02",
    max_frames: Optional[int] = None,
    ocr_lang: str = "vi",
) -> None:
    seq = KittiRawSequence(sync_path=sync_path, calib_path=calib_path, camera=camera)
    phase3_rows = _load_phase3_pose_rows(phase3_output) if phase3_output else {}
    ocr = create_ocr_backend(lang=ocr_lang)
    detector = Phase4LandmarkDetector(
        camera_fx=seq.camera_params.fx,
        camera_cx=seq.camera_params.cx,
        ocr=ocr,
    )
    db = LandmarkDatabase(sequence_id=seq.sequence_id)

    out_path = Path(output)
    observations_path = out_path.with_suffix(out_path.suffix + ".observations.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frames = 0
    observations = 0
    new_count = 0
    matched_count = 0
    class_counts: Dict[str, int] = {}

    try:
        with observations_path.open("w", encoding="utf-8") as obs_file:
            for sample in seq.iter_samples(max_frames=max_frames):
                pose = phase3_rows.get(sample.meta.frame_index)
                if pose is None:
                    pose = _pose_from_sample_xy(sample.local_xy)

                detected = detector.detect(
                    frame=sample.frame,
                    pose=pose,
                    sequence_id=seq.sequence_id,
                    frame_index=sample.meta.frame_index,
                    timestamp=sample.meta.timestamp,
                )
                frames += 1
                for obs in detected:
                    record, match = db.upsert(obs)
                    observations += 1
                    class_counts[obs.class_name] = class_counts.get(obs.class_name, 0) + 1
                    if match is None:
                        new_count += 1
                    else:
                        matched_count += 1
                    obs_file.write(
                        json.dumps(
                            {
                                "observation_id": obs.observation_id,
                                "landmark_id": record.id,
                                "match": None if match is None else match.__dict__,
                                "class": obs.class_name,
                                "timestamp": obs.timestamp,
                                "frame_index": obs.frame_index,
                                "bbox_xyxy": list(obs.bbox_xyxy),
                                "p_3D": obs.p_3D.__dict__,
                                "d_visual": obs.d_visual.__dict__,
                                "source": obs.source,
                                "attributes": obs.attributes,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    finally:
        seq.close()

    db.to_jsonl(str(out_path))
    summary = {
        "sequence": seq.sequence_id,
        "frames": frames,
        "landmarks": len(db),
        "observations": observations,
        "new_landmarks": new_count,
        "matched_observations": matched_count,
        "class_counts": class_counts,
        "output": str(out_path),
        "observations_output": str(observations_path),
        "ocr_backend": ocr.name,
        "ocr_lang": getattr(ocr, "lang", ocr_lang),
    }
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Phase 4 landmark DB complete: sequence={seq.sequence_id}, "
        f"frames={frames}, landmarks={len(db)}, observations={observations}, "
        f"output={out_path}, summary={summary_path}"
    )


def _load_phase3_pose_rows(path: Optional[str]) -> Dict[int, Pose2D]:
    if not path:
        return {}
    rows: Dict[int, Pose2D] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample = row.get("sample", {})
            frame_index = sample.get("frame_index")
            pose_data = row.get("fused_pose") or row.get("pose_local")
            if frame_index is None or pose_data is None:
                continue
            rows[int(frame_index)] = Pose2D(
                x=float(pose_data.get("x", 0.0)),
                y=float(pose_data.get("y", 0.0)),
                theta=float(pose_data.get("theta", 0.0)),
            )
    return rows


def _pose_from_sample_xy(local_xy) -> Pose2D:
    if local_xy is None:
        return Pose2D(x=0.0, y=0.0, theta=0.0)
    return Pose2D(x=float(local_xy[0]), y=float(local_xy[1]), theta=0.0)


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
    parser = argparse.ArgumentParser(description="Phase 4 visual landmark DB builder")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sequence", default=None, help="Sequence id fragment, e.g. 0001")
    parser.add_argument("--sync", default=None, help="Explicit KITTI *_sync.zip or extracted *_sync directory")
    parser.add_argument("--calib", default=None, help="Explicit KITTI calib zip/file")
    parser.add_argument("--phase3-output", default=None, help="Optional Phase 3 JSONL for fused pose")
    parser.add_argument("--camera", default="image_02", choices=["image_02", "image_03"])
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--ocr-lang", default="vi", help="PaddleOCR language code, default Vietnamese")
    parser.add_argument("--output", default=None, help="Output landmark JSONL path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sync_path, calib_path = resolve_sequence(args)
    seq_id = Path(sync_path).name.replace("_sync.zip", "").replace("_sync", "")
    output = args.output or str(Path("data") / "phase4_landmarks" / f"{seq_id}_{args.camera}.landmarks.jsonl")
    build_landmark_db(
        sync_path=sync_path,
        calib_path=calib_path,
        output=output,
        phase3_output=args.phase3_output,
        camera=args.camera,
        max_frames=args.max_frames,
        ocr_lang=args.ocr_lang,
    )


if __name__ == "__main__":
    main()
