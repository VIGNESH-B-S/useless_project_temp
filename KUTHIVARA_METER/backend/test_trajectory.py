"""Stage 3 baseline: track the visible pen endpoint in paper coordinates.

This is a diagnostic baseline.  Its length is not yet a validated ink-path
measurement because the visible barrel endpoint can be offset from the tip.
"""

import argparse
import json
import os

import cv2

from vision.paper_detector import find_calibration_frame, warp_paper
from vision.ink_estimator import DEFAULT_ML_PER_CM, estimate_ink_ml
from vision.pen_detector import build_paper_mask, detect_pen
from vision.trajectory_tracker import TrajectoryTracker
from vision.video_processor import VideoProcessor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--paper", default="A4", choices=["A4"])
    parser.add_argument("--px-per-mm", type=float, default=4.0)
    parser.add_argument("--out", default="debug_output")
    parser.add_argument("--max-gap-frames", type=int, default=3)
    parser.add_argument("--max-step-mm", type=float, default=15.0)
    parser.add_argument("--ink-ml-per-cm", type=float, default=DEFAULT_ML_PER_CM,
                        help="Known ink use of this pen in mL per centimetre")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    width_mm, height_mm = 210.0, 297.0
    vp = VideoProcessor(args.video)
    calibration_index, calibration_frame, corners = find_calibration_frame(vp)
    if corners is None:
        raise SystemExit("Paper not found; run test_paper_detection.py first.")

    _, homography, px_per_mm = warp_paper(
        calibration_frame, corners, width_mm, height_mm, args.px_per_mm
    )
    paper_mask = build_paper_mask(calibration_frame.shape, corners)
    out_size = (round(width_mm * px_per_mm), round(height_mm * px_per_mm))
    tracker = TrajectoryTracker(
        homography, px_per_mm, args.max_gap_frames, args.max_step_mm
    )

    writer = cv2.VideoWriter(
        os.path.join(args.out, "trajectory_preview.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"), vp.fps or 30.0, out_size,
    )
    final_view = None
    for frame_index, frame in vp.frames():
        result = detect_pen(frame, paper_mask)
        tracker.update(frame_index, result)
        view = cv2.warpPerspective(frame, homography, out_size)
        for start, end in tracker.segments:
            cv2.line(view, tuple(start.point_px.astype(int)), tuple(end.point_px.astype(int)),
                     (0, 255, 255), 2)
        if tracker.points:
            cv2.circle(view, tuple(tracker.points[-1].point_px.astype(int)), 5, (0, 0, 255), -1)
        cv2.putText(view, f"baseline length: {tracker.length_mm:.1f} mm", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        writer.write(view)
        final_view = view

    writer.release()
    vp.release()
    if final_view is not None:
        cv2.imwrite(os.path.join(args.out, "trajectory_full.png"), final_view)

    report = tracker.summary()
    report.update({
        "paper_calibration_frame": calibration_index,
        "ink_rate_ml_per_cm": args.ink_ml_per_cm,
        "estimated_ink_ml": estimate_ink_ml(tracker.length_mm, args.ink_ml_per_cm),
        "estimated_ink_microlitres": estimate_ink_ml(tracker.length_mm, args.ink_ml_per_cm) * 1000.0,
        "note": "Estimate uses trajectory length and the configured fixed pen rate.",
    })
    report_path = os.path.join(args.out, "ink_report.json")
    with open(report_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=2)

    print(f"Paper calibration frame: {calibration_index}")
    for key, value in report.items():
        print(f"{key}: {value:.3f}" if isinstance(value, float) else f"{key}: {value}")
    print("WARNING: baseline tracks a visible barrel endpoint, not verified ink contact.")
    print(f"Saved preview -> {os.path.join(args.out, 'trajectory_preview.mp4')}")
    print(f"Saved final overlay -> {os.path.join(args.out, 'trajectory_full.png')}")
    print(f"Saved ink report -> {report_path}")


if __name__ == "__main__":
    main()
