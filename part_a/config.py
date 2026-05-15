from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .calibration import CameraParams


SEG_POT_MODEL_PATH = "models/yolov8s_pothole.onnx"
DEFAULT_DEPTH_MODEL_PATH = "models/depth_anything_v2_vits.onnx"


@dataclass(frozen=True)
class CameraDefaults:
    fx: float = 800.0
    fy: float = 800.0
    cx: float = 640.0
    cy: float = 360.0
    width: int = 1280
    height: int = 720
    camera_height: float = 1.2
    pitch_deg: float = 5.0


DEFAULT_CAMERA = CameraDefaults()


def placeholder_camera(defaults: CameraDefaults = DEFAULT_CAMERA) -> CameraParams:
    from .calibration import CameraParams

    return CameraParams(
        fx=defaults.fx,
        fy=defaults.fy,
        cx=defaults.cx,
        cy=defaults.cy,
        width=defaults.width,
        height=defaults.height,
        h_camera=defaults.camera_height,
        pitch=np.deg2rad(defaults.pitch_deg),
    )


def warn_placeholder_camera(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=2)


def load_camera_from_args(args: argparse.Namespace) -> CameraParams:
    if args.calib:
        from .calibration import load_calibration_from_yaml

        return load_calibration_from_yaml(args.calib)
    warn_placeholder_camera(
        "Running without --calib. Metric depth/area results are approximate only."
    )
    from .calibration import CameraParams

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


def add_part_a_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--yolo", default=SEG_POT_MODEL_PATH, help="Fine-tuned YOLOv8-seg ONNX path")
    parser.add_argument("--depth", default=DEFAULT_DEPTH_MODEL_PATH, help="Depth Anything ONNX path")
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--depth-every-n", type=int, default=4, help="Run depth inference once every N processed frames")
    parser.add_argument("--severity-mode", default="area_ratio", choices=["area_ratio", "area_m2"])


def add_camera_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--calib", default=None, help="Optional camera calibration YAML")
    parser.add_argument("--fx", type=float, default=DEFAULT_CAMERA.fx)
    parser.add_argument("--fy", type=float, default=DEFAULT_CAMERA.fy)
    parser.add_argument("--cx", type=float, default=DEFAULT_CAMERA.cx)
    parser.add_argument("--cy", type=float, default=DEFAULT_CAMERA.cy)
    parser.add_argument("--width", type=int, default=DEFAULT_CAMERA.width)
    parser.add_argument("--height", type=int, default=DEFAULT_CAMERA.height)
    parser.add_argument("--camera-height", type=float, default=DEFAULT_CAMERA.camera_height)
    parser.add_argument("--pitch-deg", type=float, default=DEFAULT_CAMERA.pitch_deg)
