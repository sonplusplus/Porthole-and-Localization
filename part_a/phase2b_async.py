from __future__ import annotations
import argparse
import json
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np

from .calibration import CameraParams, load_calibration_from_yaml
from .phase2b_schema import (
    DEFAULT_DEPTH_MODEL_PATH,
    WAITING_MODEL_PATH,
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
        yolo_path: str = WAITING_MODEL_PATH,
        depth_path: str = DEFAULT_DEPTH_MODEL_PATH,
        output: Optional[str] = None,
        show: bool = False,
        max_frames: Optional[int] = None,
        queue_size: int = 2,
        drop_old_frames: bool = True,
        imgsz: int = 416,
        conf: float = 0.25,
        iou: float = 0.45,
        cam: Optional[CameraParams] = None,
    ):
        self.source = source
        self.yolo_path = yolo_path
        self.depth_path = depth_path
        self.output = output
        self.show = show
        self.max_frames = max_frames
        self.drop_old_frames = drop_old_frames
        self.frame_queue: queue.Queue[Optional[FramePacket]] = queue.Queue(maxsize=queue_size)
        self.timings: List[Phase2BTiming] = []
        self.frames_read = 0
        self.frames_dropped = 0
        self._stop = threading.Event()
        self._process = psutil.Process() if psutil is not None else None

        self.pipeline = PotholePipeline(
            yolo_path=yolo_path,
            depth_path=depth_path,
            cam=cam,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )

    def run(self) -> dict:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video source: {self.source}")

        writer = self._make_writer(cap)
        capture_thread = threading.Thread(target=self._capture_loop, args=(cap,), daemon=True)
        capture_thread.start()

        try:
            while True:
                packet = self.frame_queue.get()
                if packet is None:
                    break

                t0 = time.perf_counter()
                result = self.pipeline.process_frame(packet.frame)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                if writer is not None:
                    writer.write(result.frame)

                if self.show:
                    cv2.imshow("phase2b async pothole pipeline", result.frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        self._stop.set()
                        break

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
        finally:
            self._stop.set()
            capture_thread.join(timeout=2.0)
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
    parser.add_argument("--summary", default=None, help="Optional JSON summary path")
    parser.add_argument("--show", action="store_true", help="Show live preview")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--queue-size", type=int, default=2)
    parser.add_argument("--keep-queued-frames", action="store_true")

    parser.add_argument("--yolo", default=WAITING_MODEL_PATH, help="Fine-tuned YOLOv8-seg ONNX path")
    parser.add_argument("--depth", default=DEFAULT_DEPTH_MODEL_PATH, help="Depth Anything ONNX path")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)

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
        show=args.show,
        max_frames=args.max_frames,
        queue_size=args.queue_size,
        drop_old_frames=not args.keep_queued_frames,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        cam=load_camera(args),
    )
    summary = runner.run()
    print(json.dumps(summary, indent=2))

    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
