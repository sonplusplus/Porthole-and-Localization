import json
import math

from .events import UTurnDetector
from .schema import Pose2D


def run_check() -> dict:
    detector = UTurnDetector(window_sec=2.0, threshold_deg=150.0)
    headings_deg = [0.0, 30.0, 70.0, 110.0, 151.0, 178.0]
    timestamps = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0]
    events = []

    for timestamp, heading_deg in zip(timestamps, headings_deg):
        pose = Pose2D(x=0.0, y=0.0, theta=math.radians(heading_deg))
        event = detector.update(timestamp, pose)
        events.append(
            {
                "timestamp": timestamp,
                "heading_deg": heading_deg,
                "u_turn": event.u_turn,
                "heading_delta_deg": event.heading_delta_deg,
            }
        )

    detected = any(event["u_turn"] for event in events)
    first_detection = next((event for event in events if event["u_turn"]), None)
    return {
        "scenario": "synthetic_heading_reversal_178deg_in_2sec",
        "expected": "u_turn_detected",
        "passed": detected,
        "first_detection": first_detection,
        "events": events,
    }


def main() -> None:
    result = run_check()
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
