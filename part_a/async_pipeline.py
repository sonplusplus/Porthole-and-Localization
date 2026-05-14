from __future__ import annotations
import argparse
import json
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, TextIO, Union

import cv2
import numpy as np

from .calibration import CameraParams, load_calibration_from_yaml
from .schema import (
    DEFAULT_DEPTH_MODEL_PATH,
    SEG_POT_MODEL_PATH,
    Phase2BTiming,
    build_summary,
)
from .pipeline import PotholePipeline
try:
    import psutil
except ImportError:
    psutil = None

@dataclass
class FramePacket:
    index: int
    capture_ts: float
    frame: np.ndarray


class AsyncPartAPipeline:
    def __init__(
        self,
        source: Union[str, int],
        yolo_path: str = SEG_POT_MODEL_PATH,
        depth_path: str = DEFAULT_DEPTH_MODEL_PATH,
        output: Optional[str] = None,
        detections_output: Optional[str] = None,
        show: bool = False,
        max_frames: Optional[int] = None,
        queue_size: int = 2,
        drop_old_frames: bool = True,
        process_all_frames: bool = False,
        imgsz: int = 448,
        conf: float = 0.25,
        iou: float = 0.45,
        cam: Optional[CameraParams] = None,
        depth_every_n: int = 4,
        severity_mode: str = "area_ratio",
    ):
        self.source = source
        self.yolo_path = yolo_path
        self.depth_path = depth_path
        self.output = output
        self.detections_output = detections_output
        self.show = show
        self.max_frames = max_frames
        self.drop_old_frames = drop_old_frames
        self.process_all_frames = process_all_frames
        self.frame_queue: queue.Queue[Optional[FramePacket]] = queue.Queue(maxsize=queue_size)
        self.timings: List[Phase2BTiming] = []
        self.frames_read = 0
        self.frames_dropped = 0
        self.source_fps = 0.0
        self._detections_fp: Optional[TextIO] = None
        self._stop = threading.Event()
        self._process = psutil.Process() if psutil is not None else None

        self.pipeline = PotholePipeline(
            yolo_path=yolo_path,
            depth_path=depth_path,
            cam=cam,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            depth_every_n=depth_every_n,
            severity_mode=severity_mode,
        )

    def run(self) -> dict:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {self.source}")

        self.source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        writer = self._make_writer(cap)
        try:
            if self.detections_output:
                detections_path = Path(self.detections_output)
                detections_path.parent.mkdir(parents=True, exist_ok=True)
                with detections_path.open("w", encoding="utf-8") as fp:
                    self._detections_fp = fp
                    self._run_with_mode(cap, writer)
            else:
                self._run_with_mode(cap, writer)
        finally:
            self._stop.set()
            self._detections_fp = None
            cap.release()
            if writer is not None:
                writer.release()
            if self.show:
                cv2.destroyAllWindows()

        return build_summary(
            self.timings,
            frames_read=self.frames_read,
            frames_dropped=self.frames_dropped,
            model_path=self.yolo_path,
            depth_model_path=self.depth_path,
        ).to_dict()

    def _run_with_mode(self, cap: cv2.VideoCapture, writer: Optional[cv2.VideoWriter]) -> None:
        if self.process_all_frames:
            self._run_sequential(cap, writer)
            return

        capture_thread = threading.Thread(target=self._capture_loop, args=(cap,), daemon=True)
        capture_thread.start()
        try:
            while True:
                packet = self.frame_queue.get()
                if packet is None:
                    break
                if not self._process_packet(packet, writer):
                    break
        finally:
            capture_thread.join(timeout=2.0)

    def _run_sequential(self, cap: cv2.VideoCapture, writer: Optional[cv2.VideoWriter]) -> None:
        while not self._stop.is_set():
            if self.max_frames is not None and self.frames_read >= self.max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            packet = FramePacket(
                index=self.frames_read,
                capture_ts=time.time(),
                frame=frame,
            )
            self.frames_read += 1
            if not self._process_packet(packet, writer):
                break

    def _process_packet(self, packet: FramePacket, writer: Optional[cv2.VideoWriter]) -> bool:
        t0 = time.perf_counter()
        result = self.pipeline.process_frame(packet.frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if writer is not None:
            writer.write(result.frame)

        self._write_detections(packet, result)

        if self.show:
            cv2.imshow("phase2b async pothole pipeline", result.frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                self._stop.set()
                return False

        self.timings.append(
            Phase2BTiming(
                frame_index=packet.index,
                capture_ts=packet.capture_ts,
                end_to_end_ms=elapsed_ms,
                detect_ms=result.detect_ms,
                depth_ms=result.depth_ms,
                fps=result.fps,
                pothole_count=len(result.observations),
                queue_size=self.frame_queue.qsize(),
                memory_mb=self._memory_mb(),
            )
        )
        return True

    def _write_detections(self, packet: FramePacket, result) -> None:
        if self._detections_fp is None:
            return

        frame_h, frame_w = packet.frame.shape[:2]
        time_s = packet.index / self.source_fps if self.source_fps > 0 else None
        for det_id, obs in enumerate(result.observations):
            det = obs.detection
            metrics = obs.metrics
            record = {
                "frame_index": int(packet.index),
                "time_s": time_s,
                "capture_ts": float(packet.capture_ts),
                "detection_id": int(det_id),
                "frame_width": int(frame_w),
                "frame_height": int(frame_h),
                "bbox_xyxy": [int(v) for v in det.bbox_xyxy],
                "conf": float(det.conf),
                "cls_id": int(det.cls_id),
                "mask_area_px": int(det.area_px),
                "area_m2": float(metrics.area_m2),
                "area_ratio": float(metrics.area_ratio),
                "depth_m": float(metrics.depth_m),
                "depth_delta_m": float(metrics.depth_delta_m),
                "depth_rel": float(metrics.depth_rel),
                "severity": metrics.severity,
                "severity_idx": int(metrics.severity_idx),
                "centroid_xy_m": [
                    float(metrics.centroid_xy[0]),
                    float(metrics.centroid_xy[1]),
                ],
                "fps": float(result.fps),
                "detect_ms": float(result.detect_ms),
                "depth_ms": float(result.depth_ms),
            }
            self._detections_fp.write(json.dumps(record, ensure_ascii=True) + "\n")

    def _capture_loop(self, cap: cv2.VideoCapture) -> None:
        while not self._stop.is_set():
            if self.max_frames is not None and self.frames_read >= self.max_frames:
                break

            ok, frame = cap.read()
            if not ok:
                break

            packet = FramePacket(
                index=self.frames_read,
                capture_ts=time.time(),
                frame=frame,
            )
            self.frames_read += 1

            if self.drop_old_frames and self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

            try:
                self.frame_queue.put(packet, timeout=0.05)
            except queue.Full:
                self.frames_dropped += 1

        self.frame_queue.put(None)

    def _make_writer(self, cap: cv2.VideoCapture) -> Optional[cv2.VideoWriter]:
        if not self.output:
            return None

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(self.output, fourcc, fps, (width, height))

    def _memory_mb(self) -> Optional[float]:
        if self._process is None:
            return None
        return float(self._process.memory_info().rss / (1024 * 1024))


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
    parser = argparse.ArgumentParser(description="Phase 2B async Part A video pipeline")
    parser.add_argument("--source", required=True, help="Video path or webcam index, e.g. 0")
    parser.add_argument("--output", default=None, help="Optional output video path")
    parser.add_argument("--detections", default=None, help="Optional JSONL path for per-detection depth/area records")
    parser.add_argument("--summary", default=None, help="Optional JSON summary path")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--keep-queued-frames", action="store_true")
    parser.add_argument("--process-all-frames", action="store_true", help="Process video frames sequentially without realtime dropping")

    parser.add_argument("--yolo", default=SEG_POT_MODEL_PATH, help="Fine-tuned YOLOv8-seg ONNX path")
    parser.add_argument("--depth", default=DEFAULT_DEPTH_MODEL_PATH, help="Depth Anything ONNX path")
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


def parse_source(value: str) -> Union[str, int]:
    return int(value) if value.isdigit() else value


def main() -> None:
    args = parse_args()
    runner = AsyncPartAPipeline(
        source=parse_source(args.source),
        yolo_path=args.yolo,
        depth_path=args.depth,
        output=args.output,
        detections_output=args.detections,
        show=args.show,
        max_frames=args.max_frames,
        queue_size=args.queue_size,
        drop_old_frames=not args.keep_queued_frames,
        process_all_frames=args.process_all_frames,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        cam=load_camera(args),
        depth_every_n=args.depth_every_n,
        severity_mode=args.severity_mode,
    )
    summary = runner.run()
    print(json.dumps(summary, indent=2))

    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
