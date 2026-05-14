from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class SegmentationResult:
    mask: np.ndarray
    bbox_xyxy: Tuple[int, int, int, int]
    conf: float
    cls_id: int
    polygon_xy: Optional[np.ndarray] = None

    @property
    def area_px(self) -> int:
        return int(np.count_nonzero(self.mask))


class YOLOSegDetector:
    def __init__(
        self,
        model_path: str = "models/yolov8s_pothole.onnx",
        imgsz: int = 416,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
    ):
        path = Path(model_path)
        if model_path == "waiting":
            raise FileNotFoundError(
                "YOLO segmentation model path is still set to 'waiting'. "
                "Replace it with the Kaggle ONNX export path before running "
                "real Part A or Phase 2B inference."
            )
        if not path.exists():
            raise FileNotFoundError(
                f"YOLO segmentation model not found: {model_path}. "
                "Copy the Kaggle export to models/yolov8s_pothole.onnx "
                "or pass --yolo /path/to/best.onnx."
            )

        self.model_path = str(path)
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.device = device
        self.model = YOLO(self.model_path, task="segment")

    def predict(self, frame: np.ndarray) -> List[SegmentationResult]:
        """Run YOLOv8-seg and return binary masks in the input frame size."""
        h, w = frame.shape[:2]
        result = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )[0]

        if result.masks is None or result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy()
        polygons = getattr(result.masks, "xy", None)

        detections: List[SegmentationResult] = []
        for idx, mask in enumerate(masks):
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

            bin_mask = (mask > 0.5).astype(np.uint8)
            if np.count_nonzero(bin_mask) == 0:
                continue

            x1, y1, x2, y2 = boxes[idx]
            polygon = None
            if polygons is not None and idx < len(polygons) and len(polygons[idx]) > 0:
                polygon = np.asarray(polygons[idx], dtype=np.float32)

            detections.append(
                SegmentationResult(
                    mask=bin_mask,
                    bbox_xyxy=(
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ),
                    conf=float(confs[idx]),
                    cls_id=int(classes[idx]),
                    polygon_xy=polygon,
                )
            )
        return detections
