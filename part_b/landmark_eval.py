import argparse
import csv
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def load_observations(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["class"] = row.get("class") or row.get("class_name") or "unknown"
            rows.append(row)
    return rows


def load_ground_truth(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}

    labels: Dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        if "observation_id" not in fields or "gt_landmark_id" not in fields:
            raise ValueError("Ground-truth CSV must contain observation_id,gt_landmark_id")
        for row in reader:
            observation_id = (row.get("observation_id") or "").strip()
            gt_landmark_id = (row.get("gt_landmark_id") or "").strip()
            if observation_id and gt_landmark_id:
                labels[observation_id] = gt_landmark_id
    return labels


def evaluate_proxy(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    totals = Counter()
    matched = Counter()
    new_landmarks = Counter()
    unique_landmarks = defaultdict(set)

    for row in rows:
        class_name = str(row.get("class") or "unknown")
        totals[class_name] += 1
        landmark_id = str(row.get("landmark_id") or "")
        if landmark_id:
            unique_landmarks[class_name].add(landmark_id)
        if row.get("match") is None:
            new_landmarks[class_name] += 1
        else:
            matched[class_name] += 1

    classes = sorted(totals)
    by_class = {}
    for class_name in classes:
        total = totals[class_name]
        by_class[class_name] = {
            "observations": total,
            "matched_observations": matched[class_name],
            "new_landmarks": new_landmarks[class_name],
            "unique_landmarks": len(unique_landmarks[class_name]),
            "reidentification_rate_proxy": matched[class_name] / total if total else 0.0,
        }

    total_observations = sum(totals.values())
    total_matched = sum(matched.values())
    return {
        "observations": total_observations,
        "matched_observations": total_matched,
        "new_landmarks": sum(new_landmarks.values()),
        "unique_landmarks": sum(len(value) for value in unique_landmarks.values()),
        "reidentification_rate_proxy": total_matched / total_observations if total_observations else 0.0,
        "by_class": by_class,
        "note": "Proxy rate is matched_observations / observations; use --ground-truth for pairwise precision/recall.",
    }


def evaluate_ground_truth(rows: List[Dict[str, object]], labels: Dict[str, str]) -> Dict[str, object]:
    labelled = [row for row in rows if str(row.get("observation_id")) in labels]
    by_class_rows: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in labelled:
        by_class_rows[str(row.get("class") or "unknown")].append(row)

    by_class = {
        class_name: _pairwise_metrics(class_rows, labels)
        for class_name, class_rows in sorted(by_class_rows.items())
    }
    return {
        "labelled_observations": len(labelled),
        "all": _pairwise_metrics(labelled, labels),
        "by_class": by_class,
    }


def _pairwise_metrics(rows: List[Dict[str, object]], labels: Dict[str, str]) -> Dict[str, object]:
    tp = fp = fn = tn = 0
    for left, right in combinations(rows, 2):
        left_obs = str(left.get("observation_id"))
        right_obs = str(right.get("observation_id"))
        actual_same = labels[left_obs] == labels[right_obs]
        left_pred = str(left.get("landmark_id") or "")
        right_pred = str(right.get("landmark_id") or "")
        predicted_same = bool(left_pred and right_pred and left_pred == right_pred)
        if predicted_same and actual_same:
            tp += 1
        elif predicted_same and not actual_same:
            fp += 1
        elif not predicted_same and actual_same:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision is not None and recall is not None and (precision + recall) else None
    return {
        "pairs": tp + fp + fn + tn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Phase 4 landmark re-identification")
    parser.add_argument("--observations", required=True, help="JSONL produced by build_landmarks.py")
    parser.add_argument("--ground-truth", default=None, help="Optional CSV: observation_id,gt_landmark_id")
    parser.add_argument("--metrics", default=None, help="Optional output metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_observations(args.observations)
    labels = load_ground_truth(args.ground_truth)
    metrics = {"proxy": evaluate_proxy(rows)}
    if labels:
        metrics["ground_truth"] = evaluate_ground_truth(rows, labels)

    text = json.dumps(metrics, indent=2, ensure_ascii=False)
    print(text)
    if args.metrics:
        path = Path(args.metrics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
