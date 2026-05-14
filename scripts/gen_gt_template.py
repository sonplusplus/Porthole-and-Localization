import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Sinh GT template CSV từ detections JSONL")
    p.add_argument("--detections", required=True, help="JSONL từ part_a.benchmark --detections")
    p.add_argument("--output", required=True, help="Output CSV path")
    p.add_argument(
        "--min-conf", type=float, default=0.0,
        help="Chỉ lấy detection có conf >= ngưỡng này (mặc định: lấy hết)"
    )
    p.add_argument(
        "--sample-every", type=int, default=1,
        help="Chỉ lấy 1 detection mỗi N frame (để giảm số dòng cần điền tay)"
    )
    return p.parse_args()


def main():
    args = parse_args()
    det_path = Path(args.detections)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seen_frames = set()

    with det_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            fi = int(d["frame_index"])

            if d.get("conf", 0) < args.min_conf:
                continue
            if args.sample_every > 1 and fi % args.sample_every != 0:
                continue

            rows.append({
                "frame_index":    fi,
                "detection_id":   d.get("detection_id", 0),
                "time_s":         round(d["time_s"], 2) if d.get("time_s") is not None else "",
                "conf":           round(d.get("conf", 0), 3),
                "severity_pred":  d.get("severity", ""),
                "area_pred_m2":   round(d.get("area_m2", 0), 4),
                "depth_pred_m":   round(d.get("depth_delta_m", 0), 4),
                "bbox_xyxy":      str(d.get("bbox_xyxy", "")),
                "gt_area_m2":     "",   # A = pi × a × b  (measure in iphone L and W: a =L/2, b =W/2)
                "gt_depth_delta_m": "", 
                "notes":          "",   
            })

    if not rows:
        print("Không tìm thấy detection nào. Kiểm tra lại --detections path.")
        return

    rows.sort(key=lambda r: (r["frame_index"], r["detection_id"]))

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Đã tạo template: {out_path}  ({len(rows)} dòng)")
    print()
    print("  gt_area_m2        = π × (L/2) × (W/2)   [đo L, W của ổ gà ngoài thực tế]")
    print("  gt_depth_delta_m  = độ chênh sâu so với mặt đường xung quanh (m)")
    print("  notes             = ghi chú vật tham chiếu dùng để đo")
    print()


if __name__ == "__main__":
    main()