from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from .schema import LaneEstimate


Point = Tuple[float, float]
LaneSideMode = Literal["binary_road", "ego_offset"]


class FastHeuristicLaneDetector:
    """Fast CPU lane-side detector for latency profiling.

    This is not intended to claim B4 accuracy by itself. It is a lightweight
    OpenCV profile using color/edge masking, Hough lines, and temporal smoothing
    so Phase 3 can still run at useful FPS when the UFLDv2 Res34 model is too
    heavy for CPU.
    """

    def __init__(
        self,
        center_deadband_px: Optional[float] = None,
        center_deadband_ratio: float = 0.028,
        roi_top_ratio: float = 0.55,
        process_width: int = 640,
        min_confidence: float = 0.25,
        smoothing: float = 0.65,
        lane_side_mode: LaneSideMode = "binary_road",
        default_side: str = "right",
    ) -> None:
        self.center_deadband_px = center_deadband_px
        self.center_deadband_ratio = center_deadband_ratio
        self.roi_top_ratio = roi_top_ratio
        self.process_width = process_width
        self.min_confidence = min_confidence
        self.smoothing = smoothing
        self.lane_side_mode = lane_side_mode
        self.default_side = _normalize_binary_side(default_side)
        self.prev_lane_center: Optional[float] = None
        self.prev_lane_side: Optional[str] = None

    def estimate(self, frame: np.ndarray) -> LaneEstimate:
        h, w = frame.shape[:2]
        scale = min(1.0, self.process_width / float(w))
        if scale < 1.0:
            small = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            small = frame
        sh, sw = small.shape[:2]

        roi_mask = np.zeros((sh, sw), dtype=np.uint8)
        roi = np.array(
            [[
                (int(0.05 * sw), sh),
                (int(0.42 * sw), int(self.roi_top_ratio * sh)),
                (int(0.58 * sw), int(self.roi_top_ratio * sh)),
                (int(0.95 * sw), sh),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(roi_mask, roi, 255)

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, np.array([0, 0, 165]), np.array([180, 80, 255]))
        yellow = cv2.inRange(hsv, np.array([15, 50, 80]), np.array([40, 255, 255]))
        color_mask = cv2.bitwise_or(white, yellow)

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 70, 170)
        lane_pixels = cv2.bitwise_or(edges, color_mask)
        lane_pixels = cv2.bitwise_and(lane_pixels, roi_mask)
        lane_pixels = cv2.morphologyEx(lane_pixels, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        lines = cv2.HoughLinesP(
            lane_pixels,
            rho=1,
            theta=np.pi / 180,
            threshold=35,
            minLineLength=max(24, sw // 16),
            maxLineGap=max(28, sw // 18),
        )
        if lines is None:
            return self._estimate_legacy(frame)

        left_x: List[float] = []
        right_x: List[float] = []
        y_eval = sh * 0.80
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.35:
                continue
            length = float(np.hypot(x2 - x1, y2 - y1))
            if length < max(20, sw * 0.035):
                continue
            intercept = y1 - slope * x1
            x_eval = (y_eval - intercept) / slope
            if not (0 <= x_eval <= sw):
                continue
            if slope < 0:
                left_x.append(float(x_eval))
            else:
                right_x.append(float(x_eval))

        if not left_x or not right_x:
            return self._estimate_legacy(frame)

        left = float(np.median(left_x)) / scale
        right = float(np.median(right_x)) / scale
        if right <= left:
            return self._estimate_legacy(frame)

        raw_center = 0.5 * (left + right)
        if self.prev_lane_center is None:
            lane_center = raw_center
        else:
            lane_center = self.smoothing * self.prev_lane_center + (1.0 - self.smoothing) * raw_center
        self.prev_lane_center = lane_center

        confidence = min(1.0, (len(left_x) + len(right_x)) / 10.0)
        if confidence < self.min_confidence:
            return self._estimate_legacy(frame)

        side = self._side_from_lane_center(lane_center, w)
        return LaneEstimate(side, float(lane_center), float(confidence), len(left_x), len(right_x))

    def _unknown(self, left_count: int = 0, right_count: int = 0) -> LaneEstimate:
        if self.lane_side_mode == "binary_road":
            return LaneEstimate(
                self.prev_lane_side or self.default_side,
                self.prev_lane_center,
                0.0,
                left_count,
                right_count,
            )
        return LaneEstimate("unknown", None, 0.0, left_count, right_count)

    def _estimate_legacy(self, frame: np.ndarray) -> LaneEstimate:
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 55, 150)

        mask = np.zeros_like(edges)
        roi = np.array(
            [[
                (int(0.08 * w), h),
                (int(0.45 * w), int(self.roi_top_ratio * h)),
                (int(0.55 * w), int(self.roi_top_ratio * h)),
                (int(0.92 * w), h),
            ]],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, roi, 255)
        edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=40,
            minLineLength=max(30, w // 18),
            maxLineGap=70,
        )
        if lines is None:
            return self._unknown()

        left_x: List[float] = []
        right_x: List[float] = []
        y_eval = h * 0.78
        for x1, y1, x2, y2 in lines[:, 0]:
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            if abs(slope) < 0.30:
                continue
            intercept = y1 - slope * x1
            x_eval = (y_eval - intercept) / slope
            if not (0 <= x_eval <= w):
                continue
            if slope < 0:
                left_x.append(float(x_eval))
            else:
                right_x.append(float(x_eval))

        if not left_x or not right_x:
            return self._unknown(len(left_x), len(right_x))

        left = float(np.median(left_x))
        right = float(np.median(right_x))
        if right <= left:
            return self._unknown(len(left_x), len(right_x))

        raw_center = 0.5 * (left + right)
        if self.prev_lane_center is None:
            lane_center = raw_center
        else:
            lane_center = self.smoothing * self.prev_lane_center + (1.0 - self.smoothing) * raw_center
        self.prev_lane_center = lane_center

        confidence = min(1.0, (len(left_x) + len(right_x)) / 12.0)
        side = self._side_from_lane_center(lane_center, w)
        return LaneEstimate(side, float(lane_center), float(confidence), len(left_x), len(right_x))

    def _center_deadband(self, width: int) -> float:
        if self.center_deadband_px is not None:
            return float(self.center_deadband_px)
        return float(self.center_deadband_ratio * width)

    def _side_from_lane_center(self, lane_center: float, image_width: int) -> str:
        image_center = image_width * 0.5
        if self.lane_side_mode == "ego_offset":
            offset = image_center - lane_center
            if abs(offset) <= self._center_deadband(image_width):
                return "center"
            return "left" if offset < 0 else "right"

        side = "right" if lane_center >= image_center else "left"
        self.prev_lane_side = side
        return side


class Ufldv2OnnxLaneDetector:
    """UFLDv2 ONNX lane-side detector.

    This runtime follows the public UFLDv2 deploy convention:
    - input frame is cropped from the top, resized to train_width/train_height
    - BGR float32 image is scaled to [0, 1] and fed as NCHW
    - outputs are expected to include loc_row, loc_col, exist_row, exist_col

    By default the returned `LaneEstimate` is the binary road lane side:
    "left" or "right". The old ego-offset semantics can still be selected with
    lane_side_mode="ego_offset" for debugging.
    """

    def __init__(
        self,
        model_path: str,
        dataset: str = "culane",
        input_width: int = 1600,
        input_height: int = 320,
        crop_ratio: float = 0.6,
        num_row: int = 72,
        num_col: int = 81,
        center_deadband_px: float = 45.0,
        smoothing: float = 0.75,
        max_missing_frames: int = 2,
        lane_side_mode: LaneSideMode = "binary_road",
        default_side: str = "right",
        providers: Optional[List[str]] = None,
    ) -> None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"UFLDv2 ONNX lane model not found: {path}. "
                "Download/export a UFLDv2 CULane ONNX model and pass --lane-model."
            )

        self.dataset = dataset.lower()
        self.input_width = input_width
        self.input_height = input_height
        self.crop_ratio = crop_ratio
        self.num_row = num_row
        self.num_col = num_col
        self.center_deadband_px = center_deadband_px
        self.smoothing = smoothing
        self.max_missing_frames = max_missing_frames
        self.lane_side_mode = lane_side_mode
        self.default_side = _normalize_binary_side(default_side)
        self.prev_lane_center: Optional[float] = None
        self.prev_lane_width: Optional[float] = None
        self.prev_lane_side: Optional[str] = None
        self.missing_frames = 0
        self.row_anchor, self.col_anchor = _anchors(self.dataset, num_row, num_col)

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=providers or ["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def estimate(self, frame: np.ndarray) -> LaneEstimate:
        h, w = frame.shape[:2]
        lanes = self.lane_points(frame)
        left_idx, left_lane, right_idx, right_lane = self._ego_lane_pair(
            lanes,
            image_center=w * 0.5,
            image_width=w,
            image_height=h,
        )

        if not left_lane or not right_lane:
            return self._missing_estimate(len(left_lane), len(right_lane))

        left_x = _lane_x_at_y(left_lane, h * 0.78)
        right_x = _lane_x_at_y(right_lane, h * 0.78)
        if left_x is None or right_x is None:
            return self._missing_estimate(len(left_lane), len(right_lane))

        lane_width = right_x - left_x
        if lane_width <= 0 or not self._lane_width_is_plausible(lane_width, w):
            return self._missing_estimate(len(left_lane), len(right_lane))

        raw_center = 0.5 * (left_x + right_x)
        if self.prev_lane_center is not None and abs(raw_center - self.prev_lane_center) > max(80.0, 0.10 * w):
            return self._missing_estimate(len(left_lane), len(right_lane))

        if self.prev_lane_center is None:
            lane_center = raw_center
        else:
            lane_center = self.smoothing * self.prev_lane_center + (1.0 - self.smoothing) * raw_center
        self.prev_lane_center = lane_center
        self.prev_lane_width = lane_width
        self.missing_frames = 0

        confidence = min(1.0, (len(left_lane) + len(right_lane)) / max(self.num_row, 1))
        side = self._side_from_lane_geometry(
            lanes=lanes,
            left_idx=left_idx,
            right_idx=right_idx,
            left_x=left_x,
            right_x=right_x,
            lane_center=lane_center,
            image_width=w,
            image_height=h,
        )
        return LaneEstimate(side, float(lane_center), float(confidence), len(left_lane), len(right_lane))

    def lane_points(self, frame: np.ndarray) -> List[List[Point]]:
        h, w = frame.shape[:2]
        tensor = self._preprocess(frame)
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        pred = self._output_dict(outputs)
        return self._decode(pred, original_width=w, original_height=h)

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        cut_height = int(frame.shape[0] * (1.0 - self.crop_ratio))
        cropped = frame[cut_height:, :, :]
        resized = cv2.resize(cropped, (self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC)
        image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        return np.ascontiguousarray(np.transpose(image[None, :, :, :], (0, 3, 1, 2)))

    def _output_dict(self, outputs: List[np.ndarray]) -> Dict[str, np.ndarray]:
        by_name = {name: array for name, array in zip(self.output_names, outputs)}
        required = {"loc_row", "loc_col", "exist_row", "exist_col"}
        if required.issubset(by_name):
            return by_name
        if len(outputs) < 4:
            raise ValueError(f"UFLDv2 model must return 4 outputs; got {len(outputs)}")
        return {
            "loc_row": outputs[0],
            "loc_col": outputs[1],
            "exist_row": outputs[2],
            "exist_col": outputs[3],
        }

    def _decode(
        self,
        pred: Dict[str, np.ndarray],
        original_width: int,
        original_height: int,
    ) -> List[List[Point]]:
        loc_row = pred["loc_row"]
        loc_col = pred["loc_col"]
        exist_row = pred["exist_row"]
        exist_col = pred["exist_col"]

        num_grid_row = loc_row.shape[1]
        num_grid_col = loc_col.shape[1]
        valid_row = np.argmax(exist_row, axis=1)
        valid_col = np.argmax(exist_col, axis=1)
        max_row = np.argmax(loc_row, axis=1)
        max_col = np.argmax(loc_col, axis=1)

        lanes: List[List[Point]] = [[] for _ in range(4)]

        for lane_idx in (1, 2):
            if np.sum(valid_row[0, :, lane_idx]) <= loc_row.shape[2] / 2:
                continue
            for row_idx in range(valid_row.shape[1]):
                if not valid_row[0, row_idx, lane_idx]:
                    continue
                center = int(max_row[0, row_idx, lane_idx])
                weights_idx = np.arange(max(0, center - 1), min(num_grid_row - 1, center + 1) + 1)
                x = _soft_argmax(loc_row[0, weights_idx, row_idx, lane_idx], weights_idx)
                x = (x + 0.5) / max(num_grid_row - 1, 1) * original_width
                y = self.row_anchor[row_idx] * original_height
                lanes[lane_idx].append((float(x), float(y)))

        for lane_idx in (0, 3):
            if np.sum(valid_col[0, :, lane_idx]) <= loc_col.shape[2] / 4:
                continue
            for col_idx in range(valid_col.shape[1]):
                if not valid_col[0, col_idx, lane_idx]:
                    continue
                center = int(max_col[0, col_idx, lane_idx])
                weights_idx = np.arange(max(0, center - 1), min(num_grid_col - 1, center + 1) + 1)
                y = _soft_argmax(loc_col[0, weights_idx, col_idx, lane_idx], weights_idx)
                y = (y + 0.5) / max(num_grid_col - 1, 1) * original_height
                x = self.col_anchor[col_idx] * original_width
                lanes[lane_idx].append((float(x), float(y)))

        return lanes

    def _ego_lane_pair(
        self,
        lanes: List[List[Point]],
        image_center: float,
        image_width: int,
        image_height: int,
    ) -> Tuple[int, List[Point], int, List[Point]]:
        if len(lanes) >= 3 and lanes[1] and lanes[2]:
            left_x = _lane_x_at_y(lanes[1], image_height * 0.78)
            right_x = _lane_x_at_y(lanes[2], image_height * 0.78)
            if (
                left_x is not None
                and right_x is not None
                and left_x < image_center < right_x
                and self._lane_width_is_plausible(right_x - left_x, image_width)
            ):
                return 1, lanes[1], 2, lanes[2]

        candidates = []
        y_eval = None
        for lane in lanes:
            if lane:
                max_y = max(point[1] for point in lane)
                y_eval = max(max_y if y_eval is None else y_eval, max_y)
        if y_eval is None:
            return -1, [], -1, []

        for lane_idx, lane in enumerate(lanes):
            x = _lane_x_at_y(lane, y_eval)
            if x is None:
                continue
            candidates.append((x, lane_idx, lane))

        left = [(x, lane_idx, lane) for x, lane_idx, lane in candidates if x < image_center]
        right = [(x, lane_idx, lane) for x, lane_idx, lane in candidates if x >= image_center]
        if not left or not right:
            return -1, [], -1, []
        left_choice = max(left, key=lambda item: item[0])
        right_choice = min(
            right,
            key=lambda item: item[0],
        )
        return left_choice[1], left_choice[2], right_choice[1], right_choice[2]

    def _lane_width_is_plausible(self, width: float, image_width: int) -> bool:
        min_width = max(80.0, 0.08 * image_width)
        max_width = 0.34 * image_width
        if width < min_width or width > max_width:
            return False
        if self.prev_lane_width is None:
            return True
        return 0.55 * self.prev_lane_width <= width <= 1.70 * self.prev_lane_width

    def _missing_estimate(self, left_count: int, right_count: int) -> LaneEstimate:
        self.missing_frames += 1
        if self.lane_side_mode == "binary_road":
            side = self.prev_lane_side or self.default_side
            confidence = 0.20 if self.prev_lane_center is not None else 0.0
            return LaneEstimate(side, self.prev_lane_center, confidence, left_count, right_count)
        if self.prev_lane_center is not None and self.missing_frames <= self.max_missing_frames:
            return LaneEstimate("unknown", float(self.prev_lane_center), 0.20, left_count, right_count)
        return LaneEstimate("unknown", None, 0.0, left_count, right_count)

    def _side_from_lane_geometry(
        self,
        lanes: List[List[Point]],
        left_idx: int,
        right_idx: int,
        left_x: float,
        right_x: float,
        lane_center: float,
        image_width: int,
        image_height: int,
    ) -> str:
        image_center = image_width * 0.5
        if self.lane_side_mode == "ego_offset":
            offset = image_center - lane_center
            if abs(offset) <= self.center_deadband_px:
                return "center"
            return "left" if offset < 0 else "right"

        y_eval = image_height * 0.78
        min_gap = max(20.0, 0.03 * image_width)
        detected_x: List[float] = []
        has_left_neighbor = False
        has_right_neighbor = False
        for lane_idx, lane in enumerate(lanes):
            x = _lane_x_at_y(lane, y_eval)
            if x is None:
                continue
            detected_x.append(x)
            if lane_idx in {left_idx, right_idx}:
                continue
            if x < left_x - min_gap:
                has_left_neighbor = True
            elif x > right_x + min_gap:
                has_right_neighbor = True

        if has_left_neighbor != has_right_neighbor:
            side = "right" if has_left_neighbor else "left"
        else:
            road_center = 0.5 * (min(detected_x) + max(detected_x)) if len(detected_x) >= 3 else image_center
            if abs(lane_center - road_center) <= self.center_deadband_px:
                side = self.prev_lane_side or ("right" if lane_center >= image_center else "left")
            else:
                side = "right" if lane_center >= road_center else "left"

        self.prev_lane_side = side
        return side


def create_lane_detector(
    backend: str,
    model_path: str = "models/ufldv2_culane_res34.onnx",
    dataset: str = "culane",
    lane_side_mode: LaneSideMode = "binary_road",
):
    backend = backend.lower()
    if backend == "ufldv2":
        return Ufldv2OnnxLaneDetector(model_path=model_path, dataset=dataset, lane_side_mode=lane_side_mode)
    if backend == "heuristic":
        return FastHeuristicLaneDetector(lane_side_mode=lane_side_mode)
    raise ValueError(f"Unsupported lane backend: {backend}")


def _normalize_binary_side(value: str) -> str:
    value = value.lower().strip()
    if value not in {"left", "right"}:
        raise ValueError(f"default_side must be 'left' or 'right', got: {value!r}")
    return value


def _anchors(dataset: str, num_row: int, num_col: int) -> Tuple[np.ndarray, np.ndarray]:
    if dataset == "tusimple":
        row_anchor = np.linspace(160, 710, num_row, dtype=np.float32) / 720.0
    elif dataset == "curvelanes":
        row_anchor = np.linspace(0.4, 1.0, num_row, dtype=np.float32)
    else:
        row_anchor = np.linspace(0.42, 1.0, num_row, dtype=np.float32)
    col_anchor = np.linspace(0.0, 1.0, num_col, dtype=np.float32)
    return row_anchor, col_anchor


def _soft_argmax(logits: np.ndarray, indexes: np.ndarray) -> float:
    logits = logits.astype(np.float64)
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / max(np.sum(probs), 1e-12)
    return float(np.sum(probs * indexes.astype(np.float64)))


def _lane_x_at_y(lane: List[Point], y_eval: float) -> Optional[float]:
    if not lane:
        return None
    points = sorted(lane, key=lambda point: point[1])
    ys = np.array([point[1] for point in points], dtype=np.float64)
    xs = np.array([point[0] for point in points], dtype=np.float64)
    if len(points) == 1:
        return float(xs[0])
    y = float(np.clip(y_eval, ys[0], ys[-1]))
    return float(np.interp(y, ys, xs))
