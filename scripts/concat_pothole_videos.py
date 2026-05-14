from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concat short pothole RGB videos into one MP4.")
    parser.add_argument(
        "--input-dir",
        default="data/Pothole Videos/test/rgb",
        help="Directory containing RGB .mp4 clips.",
    )
    parser.add_argument(
        "--output",
        default="data/demo_inputs/mendeley_pothole_test_all.mp4",
        help="Output concatenated MP4 path.",
    )
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    videos = sorted(
        p for p in input_dir.glob("*.mp4")
        if p.is_file() and not p.name.startswith(".")
    )

    if not videos:
        raise FileNotFoundError(f"No .mp4 files found in: {input_dir}")

    target_size = (args.width, args.height)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, args.fps, target_size)

    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {output_path}")

    clips_written = 0
    frames_written = 0
    skipped = []

    for video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            skipped.append(str(video_path))
            continue

        local_frames = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if (frame.shape[1], frame.shape[0]) != target_size:
                frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

            writer.write(frame)
            frames_written += 1
            local_frames += 1

        cap.release()

        if local_frames > 0:
            clips_written += 1
        else:
            skipped.append(str(video_path))

    writer.release()

    duration = frames_written / args.fps if args.fps > 0 else 0.0

    print(f"Input dir: {input_dir}")
    print(f"Input clips: {len(videos)}")
    print(f"Clips written: {clips_written}")
    print(f"Frames written: {frames_written}")
    print(f"Duration: {duration:.2f}s")
    print(f"Output: {output_path}")

    if skipped:
        print("Skipped files:")
        for item in skipped:
            print(item)


if __name__ == "__main__":
    main()
