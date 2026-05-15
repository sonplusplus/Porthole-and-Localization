from __future__ import annotations

import argparse
import json
from pathlib import Path

from .calibration import CameraParams
from .async_pipeline import AsyncPartAPipeline, parse_source
from .config import add_camera_args, add_part_a_model_args, load_camera_from_args

def load_camera(args: argparse.Namespace) -> CameraParams:
    return load_camera_from_args(args)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2B FPS/latency benchmark")
    parser.add_argument("--source", required=True, help="Video path or webcam index, e.g. 0")
    parser.add_argument("--summary", default="data/phase2b_outputs/benchmark_summary.json")
    parser.add_argument("--output", default=None, help="Optional rendered benchmark video")
    parser.add_argument("--detections", default=None, help="Optional JSONL path for per-detection depth/area records")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--process-all-frames", action="store_true", help="Process frames sequentially without realtime dropping")

    add_part_a_model_args(parser)
    add_camera_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runner = AsyncPartAPipeline(
        source=parse_source(args.source),
        yolo_path=args.yolo,
        depth_path=args.depth,
        output=args.output,
        detections_output=args.detections,
        show=False,
        max_frames=args.max_frames,
        queue_size=args.queue_size,
        process_all_frames=args.process_all_frames,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        cam=load_camera(args),
        depth_every_n=args.depth_every_n,
        severity_mode=args.severity_mode,
    )
    summary = runner.run()

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
