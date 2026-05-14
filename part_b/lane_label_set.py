import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2

from .kitti import KittiRawSequence, discover_kitti_sequences
from .lane import Ufldv2OnnxLaneDetector
from .metrics import load_rows


def export_label_set(
    sync_path: str,
    calib_path: Optional[str],
    phase3_output: str,
    output_dir: str,
    labels_csv: str,
    camera: str = "image_02",
    step: int = 5,
    max_frames: Optional[int] = None,
    lane_model: str = "models/ufldv2_culane_res34.onnx",
    lane_dataset: str = "culane",
) -> None:
    rows = load_rows(phase3_output)
    rows_by_frame = {
        int(row.get("sample", {}).get("frame_index", index)): row
        for index, row in enumerate(rows)
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label_path = Path(labels_csv)
    label_path.parent.mkdir(parents=True, exist_ok=True)

    seq = KittiRawSequence(sync_path=sync_path, calib_path=calib_path, camera=camera)
    lane_debugger = Ufldv2OnnxLaneDetector(model_path=lane_model, dataset=lane_dataset) if lane_model else None
    exported: List[Dict[str, Any]] = []
    try:
        for sample in seq.iter_samples(max_frames=max_frames):
            frame_index = sample.meta.frame_index
            if frame_index % max(step, 1) != 0:
                continue
            row = rows_by_frame.get(frame_index)
            if row is None:
                continue

            raw_path = out_dir / f"frame_{frame_index:06d}.jpg"
            cv2.imwrite(str(raw_path), sample.frame)

            frame = sample.frame.copy()
            lane_points = lane_debugger.lane_points(sample.frame) if lane_debugger is not None else None
            _draw_lane_hint(frame, row, lane_points=lane_points)
            image_path = out_dir / f"frame_{frame_index:06d}.jpg"
            overlay_path = out_dir / f"frame_{frame_index:06d}_overlay.jpg"
            cv2.imwrite(str(overlay_path), frame)
            exported.append(
                {
                    "frame_index": frame_index,
                    "image_path": str(raw_path),
                    "overlay_path": str(overlay_path),
                    "predicted_lane": row.get("lane", {}).get("lane_side", "unknown"),
                    "lane_side": "",
                    "notes": "",
                }
            )
    finally:
        seq.close()

    with label_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_index", "image_path", "overlay_path", "predicted_lane", "lane_side", "notes"],
        )
        writer.writeheader()
        writer.writerows(exported)

    print(f"Exported {len(exported)} label frames to {out_dir}")
    print(f"Saved label CSV template: {label_path}")


def _draw_lane_hint(frame, row: Dict[str, Any], lane_points=None) -> None:
    lane = row.get("lane", {})
    frame_index = int(row.get("sample", {}).get("frame_index", 0))
    predicted = lane.get("lane_side", "unknown")
    confidence = float(lane.get("confidence") or 0.0)
    lane_center = lane.get("lane_center_px")

    if lane_points is not None:
        colors = [(60, 180, 255), (80, 220, 80), (70, 120, 255), (220, 80, 220)]
        for lane_index, lane in enumerate(lane_points):
            if len(lane) < 2:
                continue
            pixels = [(int(x), int(y)) for x, y in lane]
            for p0, p1 in zip(pixels, pixels[1:]):
                cv2.line(frame, p0, p1, colors[lane_index % len(colors)], 3)

    if lane_center is not None:
        x = int(float(lane_center))
        cv2.line(frame, (x, int(frame.shape[0] * 0.52)), (x, frame.shape[0] - 1), (220, 60, 220), 2)
    cv2.line(
        frame,
        (frame.shape[1] // 2, int(frame.shape[0] * 0.52)),
        (frame.shape[1] // 2, frame.shape[0] - 1),
        (255, 255, 255),
        1,
    )
    text = f"frame {frame_index} | predicted={predicted} conf={confidence:.2f}"
    cv2.rectangle(frame, (12, 12), (12 + 620, 48), (20, 20, 20), -1)
    cv2.putText(frame, text, (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (245, 245, 245), 2, cv2.LINE_AA)


def resolve_sequence(args: argparse.Namespace):
    if args.sync:
        return args.sync, args.calib

    refs = discover_kitti_sequences(args.data_root)
    if not refs:
        raise FileNotFoundError(f"No KITTI *_sync.zip or *_sync directories found under {args.data_root}")

    if args.sequence:
        for ref in refs:
            if args.sequence in ref.sequence_id:
                return str(ref.sync_path), str(ref.calib_path) if ref.calib_path else None
        raise FileNotFoundError(f"Could not find sequence matching: {args.sequence}")

    ref = refs[0]
    return str(ref.sync_path), str(ref.calib_path) if ref.calib_path else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export frame images and CSV template for lane-side labeling")
    parser.add_argument("--phase3-output", required=True, help="Phase 3 JSONL output")
    parser.add_argument("--output-dir", required=True, help="Directory for sampled frame JPGs")
    parser.add_argument("--labels-csv", required=True, help="CSV template to write")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sequence", default=None, help="Sequence id fragment, e.g. 0001 or 0056")
    parser.add_argument("--sync", default=None, help="Explicit KITTI *_sync.zip or extracted *_sync directory")
    parser.add_argument("--calib", default=None, help="Explicit KITTI calib zip/file")
    parser.add_argument("--camera", default="image_02", choices=["image_02", "image_03"])
    parser.add_argument("--step", type=int, default=5, help="Export every Nth frame")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--lane-model", default="models/ufldv2_culane_res34.onnx", help="Optional UFLDv2 model for overlay polylines")
    parser.add_argument("--lane-dataset", default="culane", choices=["culane", "tusimple", "curvelanes"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sync_path, calib_path = resolve_sequence(args)
    export_label_set(
        sync_path=sync_path,
        calib_path=calib_path,
        phase3_output=args.phase3_output,
        output_dir=args.output_dir,
        labels_csv=args.labels_csv,
        camera=args.camera,
        step=args.step,
        max_frames=args.max_frames,
        lane_model=args.lane_model,
        lane_dataset=args.lane_dataset,
    )


if __name__ == "__main__":
    main()
