from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np

from .calibration import CameraParams, load_calibration_from_yaml
from .async_pipeline import AsyncPartAPipeline, parse_source
from .schema import DEFAULT_DEPTH_MODEL_PATH, SEG_POT_MODEL_PATH

def load_camera(args: argparse.Namespace) -> CameraParams:
    if args.calib:
        return load_calibration_from_yaml(args.calib)
    warnings.warn(
        "Running without --calib. Metric depth/area results are approximate only.",
        UserWarning,
        stacklevel=2,
    )
    return CameraParams(
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        width=args.width,
        height=args.height,
        h_camera=args.camera_height,
        pitch=np.deg2rad(args.pitch_deg),
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2B FPS/latency benchmark")
    parser.add_argument("--source", required=True, help="Video path or webcam index, e.g. 0")
    parser.add_argument("--summary", default="data/phase2b_outputs/benchmark_summary.json")
    parser.add_argument("--output", default=None, help="Optional rendered benchmark video")
    parser.add_argument("--detections", default=None, help="Optional JSONL path for per-detection depth/area records")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--process-all-frames", action="store_true", help="Process frames sequentially without realtime dropping")

    parser.add_argument("--yolo", default=SEG_POT_MODEL_PATH, help="Fine-tuned YOLOv8-seg ONNX path")
    parser.add_argument("--depth", default=DEFAULT_DEPTH_MODEL_PATH, help="Depth Anything ONNX path")
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--depth-every-n", type=int, default=4, help="Run depth inference once every N processed frames")
    parser.add_argument("--severity-mode", default="area_ratio", choices=["area_ratio", "area_m2"])

    parser.add_argument("--calib", default=None)
    parser.add_argument("--fx", type=float, default=800.0)
    parser.add_argument("--fy", type=float, default=800.0)
    parser.add_argument("--cx", type=float, default=640.0)
    parser.add_argument("--cy", type=float, default=360.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--camera-height", type=float, default=1.2)
    parser.add_argument("--pitch-deg", type=float, default=5.0)
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
