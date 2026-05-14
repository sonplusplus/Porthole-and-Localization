from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part A on real demo videos as a repeatable batch.")
    parser.add_argument("--input-dir", default="data/demo_inputs")
    parser.add_argument("--output-dir", default="data/phase2b_outputs/real_demos")
    parser.add_argument("--calib-dir", default="data/calibration")
    parser.add_argument("--pattern", default="vid*.mp4")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--depth-every-n", type=int, default=4)
    parser.add_argument("--severity-mode", default="area_ratio", choices=["area_ratio", "area_m2"])
    parser.add_argument("--realtime", action="store_true", help="Allow frame dropping instead of processing every frame.")
    parser.add_argument("--no-render", action="store_true", help="Skip overlay MP4 output.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--manifest", default=None, help="Optional manifest JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    calib_dir = Path(args.calib_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(input_dir.glob(args.pattern))
    if not videos:
        raise FileNotFoundError(f"No videos matching {args.pattern!r} under {input_dir}")

    manifest: List[Dict[str, Any]] = []
    for video in videos:
        stem = video.stem
        summary_path = output_dir / f"{stem}_summary.json"
        detections_path = output_dir / f"{stem}_detections.jsonl"
        overlay_path = output_dir / f"{stem}_overlay.mp4"
        calib_path = find_calibration(calib_dir, stem)

        if args.skip_existing and summary_path.exists() and detections_path.exists():
            manifest.append(load_manifest_row(video, summary_path, detections_path, overlay_path, calib_path, skipped=True))
            continue

        cmd = [
            sys.executable,
            "-B",
            "-m",
            "part_a.benchmark",
            "--source",
            str(video),
            "--summary",
            str(summary_path),
            "--detections",
            str(detections_path),
            "--imgsz",
            str(args.imgsz),
            "--conf",
            str(args.conf),
            "--iou",
            str(args.iou),
            "--depth-every-n",
            str(args.depth_every_n),
            "--severity-mode",
            args.severity_mode,
        ]
        if calib_path is not None:
            cmd.extend(["--calib", str(calib_path)])
        if args.max_frames is not None:
            cmd.extend(["--max-frames", str(args.max_frames)])
        if not args.realtime:
            cmd.append("--process-all-frames")
        if not args.no_render:
            cmd.extend(["--output", str(overlay_path)])

        print("Running:", " ".join(cmd))
        completed = subprocess.run(cmd, check=False)
        row = load_manifest_row(video, summary_path, detections_path, overlay_path, calib_path, skipped=False)
        row["returncode"] = completed.returncode
        manifest.append(row)
        if completed.returncode != 0:
            break

    manifest_path = Path(args.manifest) if args.manifest else output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest: {manifest_path}")

    failed = [row for row in manifest if row.get("returncode", 0) != 0]
    if failed:
        raise SystemExit(f"{len(failed)} demo run(s) failed; see manifest for details.")


def find_calibration(calib_dir: Path, stem: str) -> Optional[Path]:
    exact = calib_dir / f"{stem}.yaml"
    if exact.exists():
        return exact
    matches = sorted(calib_dir.glob(f"{stem}_*.yaml"))
    return matches[0] if matches else None


def load_manifest_row(
    video: Path,
    summary_path: Path,
    detections_path: Path,
    overlay_path: Path,
    calib_path: Optional[Path],
    skipped: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "video": str(video),
        "calib": str(calib_path) if calib_path else None,
        "summary": str(summary_path),
        "detections": str(detections_path),
        "overlay": str(overlay_path) if overlay_path.exists() else None,
        "skipped": skipped,
    }
    if summary_path.exists():
        row["metrics"] = json.loads(summary_path.read_text(encoding="utf-8"))
    if detections_path.exists():
        row["detection_records"] = count_lines(detections_path)
    return row


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


if __name__ == "__main__":
    main()
