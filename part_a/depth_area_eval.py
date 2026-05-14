from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


BBox = Tuple[float, float, float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Part A depth/area predictions against a small manual GT CSV.")
    parser.add_argument("--detections", required=True, help="JSONL from part_a.benchmark --detections")
    parser.add_argument("--ground-truth", required=True, help="CSV with frame_index plus GT area/depth columns")
    parser.add_argument("--metrics", default=None, help="Output metrics JSON")
    parser.add_argument("--matches", default=None, help="Optional matched rows JSONL for audit")
    parser.add_argument("--min-iou", type=float, default=0.30, help="Minimum bbox IoU when matching by bbox_xyxy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detections = load_detections(args.detections)
    ground_truth = load_ground_truth(args.ground_truth)
    matched, unmatched = match_rows(ground_truth, detections, min_iou=args.min_iou)
    metrics = compute_metrics(matched, unmatched)

    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.metrics:
        path = Path(args.metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    if args.matches:
        path = Path(args.matches)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for row in matched:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_detections(path: str) -> Dict[int, List[Dict[str, Any]]]:
    by_frame: Dict[int, List[Dict[str, Any]]] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame_index = int(row["frame_index"])
            by_frame.setdefault(frame_index, []).append(row)
    return by_frame


def load_ground_truth(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "frame_index" not in fields:
            raise ValueError("Ground-truth CSV must contain frame_index")
        if "gt_area_m2" not in fields and "gt_depth_delta_m" not in fields:
            raise ValueError("Ground-truth CSV must contain gt_area_m2 and/or gt_depth_delta_m")
        for raw in reader:
            rows.append(raw)
    return rows


def match_rows(
    ground_truth: Sequence[Dict[str, Any]],
    detections_by_frame: Dict[int, List[Dict[str, Any]]],
    min_iou: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    matched: List[Dict[str, Any]] = []
    unmatched: List[Dict[str, Any]] = []
    for gt in ground_truth:
        frame_index = int(gt["frame_index"])
        candidates = detections_by_frame.get(frame_index, [])
        det = select_detection(gt, candidates, min_iou=min_iou)
        if det is None:
            unmatched.append({"frame_index": frame_index, "reason": "no_matching_detection", "ground_truth": gt})
            continue
        matched.append(build_match_row(gt, det))
    return matched, unmatched


def select_detection(gt: Dict[str, Any], candidates: Sequence[Dict[str, Any]], min_iou: float) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    detection_id = parse_optional_int(gt.get("detection_id"))
    if detection_id is not None:
        for det in candidates:
            if int(det.get("detection_id", -1)) == detection_id:
                return det
        return None

    gt_bbox = parse_bbox(gt)
    if gt_bbox is not None:
        scored = [(_iou(gt_bbox, tuple(det["bbox_xyxy"])), det) for det in candidates if "bbox_xyxy" in det]
        if not scored:
            return None
        score, det = max(scored, key=lambda item: item[0])
        return det if score >= min_iou else None

    if len(candidates) == 1:
        return candidates[0]
    return max(candidates, key=lambda det: float(det.get("conf", 0.0)))


def build_match_row(gt: Dict[str, Any], det: Dict[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "frame_index": int(gt["frame_index"]),
        "detection_id": int(det.get("detection_id", 0)),
        "bbox_xyxy": det.get("bbox_xyxy"),
        "conf": det.get("conf"),
        "pred_area_m2": det.get("area_m2"),
        "pred_depth_delta_m": det.get("depth_delta_m"),
        "gt_area_m2": parse_optional_float(gt.get("gt_area_m2")),
        "gt_depth_delta_m": parse_optional_float(gt.get("gt_depth_delta_m")),
    }
    add_error(row, "area_m2")
    add_error(row, "depth_delta_m")
    return row


def add_error(row: Dict[str, Any], metric: str) -> None:
    pred = parse_optional_float(row.get(f"pred_{metric}"))
    gt = parse_optional_float(row.get(f"gt_{metric}"))
    if pred is None or gt is None:
        return
    abs_error = abs(pred - gt)
    row[f"{metric}_abs_error"] = abs_error
    row[f"{metric}_pct_error"] = (abs_error / abs(gt)) if abs(gt) > 1e-9 else None


def compute_metrics(matched: Sequence[Dict[str, Any]], unmatched: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "gt_count": len(matched) + len(unmatched),
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "area_m2": summarize_metric(matched, "area_m2"),
        "depth_delta_m": summarize_metric(matched, "depth_delta_m"),
        "unmatched": list(unmatched),
    }


def summarize_metric(rows: Sequence[Dict[str, Any]], metric: str) -> Dict[str, Optional[float]]:
    abs_errors = [parse_optional_float(row.get(f"{metric}_abs_error")) for row in rows]
    pct_errors = [parse_optional_float(row.get(f"{metric}_pct_error")) for row in rows]
    abs_clean = [value for value in abs_errors if value is not None and math.isfinite(value)]
    pct_clean = [value for value in pct_errors if value is not None and math.isfinite(value)]
    return {
        "count": len(abs_clean),
        "mae": float(mean(abs_clean)) if abs_clean else None,
        "median_abs_error": float(median(abs_clean)) if abs_clean else None,
        "mape": float(mean(pct_clean)) if pct_clean else None,
        "median_pct_error": float(median(pct_clean)) if pct_clean else None,
        "p95_pct_error": percentile(pct_clean, 0.95),
    }


def parse_optional_int(value: Any) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    return int(float(str(value).strip()))


def parse_optional_float(value: Any) -> Optional[float]:
    if value is None or str(value).strip() == "":
        return None
    return float(str(value).strip())


def parse_bbox(row: Dict[str, Any]) -> Optional[BBox]:
    raw = row.get("bbox_xyxy")
    if raw:
        if isinstance(raw, str):
            value = raw.strip()
            if value.startswith("["):
                items = json.loads(value)
            else:
                items = [item.strip() for item in value.split(",")]
        else:
            items = raw
        if len(items) != 4:
            raise ValueError(f"bbox_xyxy must have 4 values: {raw}")
        return tuple(float(item) for item in items)  # type: ignore[return-value]

    keys = ("x1", "y1", "x2", "y2")
    if all(row.get(key) not in (None, "") for key in keys):
        return tuple(float(row[key]) for key in keys)  # type: ignore[return-value]
    return None


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def percentile(values: Iterable[float], q: float) -> Optional[float]:
    clean = sorted(float(value) for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return float(clean[lower] * (1.0 - weight) + clean[upper] * weight)


if __name__ == "__main__":
    main()
