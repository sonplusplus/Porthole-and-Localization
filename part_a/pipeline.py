import argparse
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import cv2
import numpy as np
from .calibration import CameraParams, IPMTransformer, load_calibration_from_yaml
from .depth import DepthEstimator, PotholeMetrics
from .detect import SegmentationResult, YOLOSegDetector


SEG_POT_MODEL_PATH = "models/yolov8s_pothole.onnx"
DEPTH_MODEL_PATH = "models/depth_anything_v2_vits.onnx"


@dataclass
class PotholeObservation:
    detection: SegmentationResult
    metrics: PotholeMetrics


@dataclass
class PipelineOutput:
    frame: np.ndarray
    observations: List[PotholeObservation]
    fps: float
    detect_ms: float
    depth_ms: float

class PotholePipeline:
    """End-to-end Part A pipeline: YOLOv8-seg mask -> depth/IPM metrics."""

    def __init__(
        self,
        yolo_path: str = SEG_POT_MODEL_PATH,
        depth_path: str = DEPTH_MODEL_PATH,
        cam: Optional[CameraParams] = None,
        imgsz: int = 448,
        conf: float = 0.25,
        iou: float = 0.45,
        depth_every_n: int = 4,
        severity_mode: str = "area_ratio",
    ):
        if cam is None:
            warnings.warn(
                "No CameraParams provided to PotholePipeline. Using placeholder values "
                "(fx=800, fy=800, cx=640, cy=360, h_camera=1.2m, pitch=5deg). "
                "Metric area_m2 and depth_m outputs will be unreliable for real cameras. "
                "Pass --calib or construct CameraParams from measured camera specs.",
                UserWarning,
                stacklevel=2,
            )
            self.cam = CameraParams(
                fx=800,
                fy=800,
                cx=640,
                cy=360,
                width=1280,
                height=720,
                h_camera=1.2,
                pitch=np.deg2rad(5),
            )
        else:
            self.cam = cam
        self.ipm = IPMTransformer(self.cam)
        self.depth_every_n = max(1, int(depth_every_n))
        self.severity_mode = severity_mode
        self._frame_index = 0
        self._cached_depth_metric: Optional[np.ndarray] = None
        self.detector = YOLOSegDetector(
            model_path=yolo_path,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            device="cpu",
        )
        self.depth = DepthEstimator(depth_path, self.cam, self.ipm)

    def process_frame(self, frame: np.ndarray) -> PipelineOutput:
        frame_index = self._frame_index
        self._frame_index += 1
        t0 = time.perf_counter()

        t_det0 = time.perf_counter()
        detections = self.detector.predict(frame)
        detect_ms = (time.perf_counter() - t_det0) * 1000

        observations: List[PotholeObservation] = []
        depth_ms = 0.0
        depth_metric = None

        # Depth is expensive, so run it only when segmentation finds potholes
        # and refresh the metric map every N frames.
        if detections:
            refresh_depth = (
                self._cached_depth_metric is None
                or self._cached_depth_metric.shape[:2] != frame.shape[:2]
                or frame_index % self.depth_every_n == 0
            )
            if refresh_depth:
                t_depth0 = time.perf_counter()
                depth_metric = self.depth.infer_metric(frame)
                depth_ms = (time.perf_counter() - t_depth0) * 1000
                self._cached_depth_metric = depth_metric
            else:
                depth_metric = self._cached_depth_metric

            for det in detections:
                metrics = self.depth.estimate_pothole(
                    frame,
                    det.mask,
                    depth_metric,
                    severity_mode=self.severity_mode,
                )
                observations.append(PotholeObservation(det, metrics))

        out = draw_observations(frame, observations)
        fps = 1.0 / max(time.perf_counter() - t0, 1e-6)
        draw_header(out, len(observations), fps, detect_ms, depth_ms)

        return PipelineOutput(
            frame=out,
            observations=observations,
            fps=fps,
            detect_ms=detect_ms,
            depth_ms=depth_ms,
        )


def draw_observations(
    frame: np.ndarray,
    observations: List[PotholeObservation],
) -> np.ndarray:
    out = frame.copy()
    overlay = out.copy()

    colors = {
        "minor": (0, 220, 0),
        "moderate": (0, 165, 255),
        "severe": (0, 0, 255),
    }

    for obs in observations:
        det = obs.detection
        metrics = obs.metrics
        color = colors.get(metrics.severity, (255, 255, 255))

        overlay[det.mask.astype(bool)] = (
            0.35 * overlay[det.mask.astype(bool)] + 0.65 * np.array(color)
        ).astype(np.uint8)

        x1, y1, x2, y2 = det.bbox_xyxy
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if det.polygon_xy is not None and len(det.polygon_xy) >= 3:
            pts = det.polygon_xy.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], isClosed=True, color=color, thickness=2)

        label = (
            f"{metrics.severity} {det.conf:.2f} | "
            f"area {metrics.area_m2:.3f}m2 | rel {metrics.depth_rel:.2f}"
        )
        draw_label(out, label, x1, max(20, y1 - 8), color)

    return cv2.addWeighted(overlay, 0.45, out, 0.55, 0)


def draw_header(
    frame: np.ndarray,
    count: int,
    fps: float,
    detect_ms: float,
    depth_ms: float,
) -> None:
    text = f"potholes:{count}  fps:{fps:.1f}  yolo:{detect_ms:.0f}ms  depth:{depth_ms:.0f}ms"
    cv2.rectangle(frame, (8, 8), (560, 42), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_label(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: Tuple[int, int, int],
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    x = max(0, min(x, frame.shape[1] - tw - 8))
    y = max(th + 6, y)
    cv2.rectangle(frame, (x, y - th - 8), (x + tw + 8, y + 4), (0, 0, 0), -1)
    cv2.putText(
        frame,
        text,
        (x + 4, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


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


def run_image(pipeline: PotholePipeline, source: str, output: Optional[str]) -> None:
    frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    result = pipeline.process_frame(frame)
    print_observations(result)

    if output:
        cv2.imwrite(output, result.frame)
        print(f"Saved: {output}")


def run_video(
    pipeline: PotholePipeline,
    source: str,
    output: Optional[str],
    show: bool,
) -> None:
    cap = cv2.VideoCapture(0 if source == "0" else source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {source}")

    writer = None
    if output:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = pipeline.process_frame(frame)
            if writer:
                writer.write(result.frame)

            if show:
                cv2.imshow("pothole pipeline", result.frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved: {output}")
        if show:
            cv2.destroyAllWindows()


def print_observations(result: PipelineOutput) -> None:
    print(
        f"fps={result.fps:.2f}, yolo={result.detect_ms:.1f}ms, "
        f"depth={result.depth_ms:.1f}ms, potholes={len(result.observations)}"
    )
    for i, obs in enumerate(result.observations, start=1):
        m = obs.metrics
        d = obs.detection
        print(
            f"#{i}: conf={d.conf:.3f}, severity={m.severity}, "
            f"area={m.area_m2:.4f}m2, depth_rel={m.depth_rel:.3f}, "
            f"centroid=({m.centroid_xy[0]:.2f}, {m.centroid_xy[1]:.2f})"
        )


def is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8-seg + Depth Anything pothole demo")
    parser.add_argument("--source", required=True, help="Image/video path, or 0 for webcam")
    parser.add_argument("--output", default=None, help="Output image/video path")
    parser.add_argument("--show", action="store_true", help="Show live window for video/webcam")

    parser.add_argument("--yolo", default=SEG_POT_MODEL_PATH, help="Fine-tuned YOLOv8-seg .onnx/.pt")
    parser.add_argument("--depth", default=DEPTH_MODEL_PATH, help="Depth Anything ONNX model")
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--depth-every-n", type=int, default=4, help="Run depth inference once every N processed frames")
    parser.add_argument("--severity-mode", default="area_ratio", choices=["area_ratio", "area_m2"])

    parser.add_argument("--calib", default=None, help="Optional camera calibration YAML")
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
    cam = load_camera(args)
    pipeline = PotholePipeline(
        yolo_path=args.yolo,
        depth_path=args.depth,
        cam=cam,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        depth_every_n=args.depth_every_n,
        severity_mode=args.severity_mode,
    )

    if is_image_path(args.source):
        run_image(pipeline, args.source, args.output)
    else:
        run_video(pipeline, args.source, args.output, args.show)


if __name__ == "__main__":
    main()
