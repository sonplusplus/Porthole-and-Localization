from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a reproducible Part B GPS-loss handover artifact.")
    parser.add_argument("--sequence", default="0001")
    parser.add_argument("--max-frames", type=int, default=80)
    parser.add_argument("--lane-backend", default="heuristic", choices=["heuristic", "ufldv2"])
    parser.add_argument("--gps-loss-start", type=int, default=20)
    parser.add_argument("--gps-loss-end", type=int, default=55)
    parser.add_argument("--gps-loss-degraded-frames", type=int, default=3)
    parser.add_argument("--landmark-db", default=None)
    parser.add_argument("--landmark-every-n", type=int, default=5)
    parser.add_argument("--output", default="data/phase3_outputs/phase5_handover_latest.jsonl")
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--plot-dir", default="data/phase3_outputs/plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    metrics = Path(args.metrics) if args.metrics else output.with_suffix(".metrics.json")

    pipeline_cmd = [
        sys.executable,
        "-B",
        "-m",
        "part_b.pipeline",
        "--sequence",
        args.sequence,
        "--max-frames",
        str(args.max_frames),
        "--lane-backend",
        args.lane_backend,
        "--gps-loss-start",
        str(args.gps_loss_start),
        "--gps-loss-end",
        str(args.gps_loss_end),
        "--gps-loss-degraded-frames",
        str(args.gps_loss_degraded_frames),
        "--output",
        str(output),
    ]
    if args.landmark_db:
        pipeline_cmd.extend(["--landmark-db", args.landmark_db, "--landmark-every-n", str(args.landmark_every_n)])

    run(pipeline_cmd)
    run(
        [
            sys.executable,
            "-B",
            "-m",
            "part_b.metrics",
            "--input",
            str(output),
            "--metrics",
            str(metrics),
            "--plot-dir",
            args.plot_dir,
        ]
    )
    print(f"Saved handover output: {output}")
    print(f"Saved handover metrics: {metrics}")


def run(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
