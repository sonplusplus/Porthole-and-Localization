import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


VALID_LABELS = {"left", "right", "center", "unknown"}


def load_predictions(path: str) -> Dict[int, str]:
    predictions: Dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            frame_index = int(row.get("sample", {}).get("frame_index"))
            lane_side = str(row.get("lane", {}).get("lane_side", "unknown"))
            predictions[frame_index] = lane_side if lane_side in VALID_LABELS else "unknown"
    return predictions


def load_labels(path: str) -> Dict[int, str]:
    labels: Dict[int, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "frame_index" not in (reader.fieldnames or []) or "lane_side" not in (reader.fieldnames or []):
            raise ValueError("Label CSV must contain columns: frame_index,lane_side")
        for row in reader:
            frame_index = int(row["frame_index"])
            lane_side = row["lane_side"].strip().lower()
            if not lane_side:
                continue
            if lane_side not in VALID_LABELS:
                raise ValueError(f"Invalid lane_side at frame {frame_index}: {lane_side}")
            labels[frame_index] = lane_side
    return labels


def evaluate(predictions: Dict[int, str], labels: Dict[int, str]) -> Dict[str, object]:
    matched: List[Tuple[int, str, str]] = []
    missing = []
    for frame_index, expected in labels.items():
        predicted = predictions.get(frame_index)
        if predicted is None:
            missing.append(frame_index)
            continue
        matched.append((frame_index, expected, predicted))

    correct = sum(1 for _, expected, predicted in matched if expected == predicted)
    confusion = Counter((expected, predicted) for _, expected, predicted in matched)
    per_class = {}
    for label in sorted(VALID_LABELS):
        total = sum(1 for _, expected, _ in matched if expected == label)
        hits = sum(1 for _, expected, predicted in matched if expected == label and predicted == label)
        per_class[label] = {
            "total": total,
            "correct": hits,
            "accuracy": (hits / total) if total else None,
        }

    return {
        "label_count": len(labels),
        "matched_count": len(matched),
        "missing_predictions": missing,
        "accuracy": (correct / len(matched)) if matched else 0.0,
        "correct": correct,
        "incorrect": len(matched) - correct,
        "per_class": per_class,
        "confusion": [
            {"expected": expected, "predicted": predicted, "count": count}
            for (expected, predicted), count in sorted(confusion.items())
        ],
    }


def write_label_template(output: str, frame_indexes: Iterable[int]) -> None:
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_index", "lane_side"])
        writer.writeheader()
        for frame_index in frame_indexes:
            writer.writerow({"frame_index": frame_index, "lane_side": ""})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 3 lane-side predictions against manual labels")
    parser.add_argument("--phase3-output", required=True, help="Phase 3 JSONL output")
    parser.add_argument("--labels", help="CSV with columns frame_index,lane_side")
    parser.add_argument("--metrics", help="Output metrics JSON")
    parser.add_argument("--write-template", help="Write a CSV label template from the JSONL frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.phase3_output)
    if args.write_template:
        write_label_template(args.write_template, sorted(predictions))
        print(f"Saved label template: {args.write_template}")
        return

    if not args.labels:
        raise ValueError("--labels is required unless --write-template is used")

    labels = load_labels(args.labels)
    metrics = evaluate(predictions, labels)
    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.metrics:
        path = Path(args.metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
