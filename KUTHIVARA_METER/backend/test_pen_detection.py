"""
test_pen_detection.py

Stage 2 test: pen blob detection.

Runs on top of Stage 1 (paper detection). Reuses the locked paper
corners to build a search mask, then runs the pen detector on every
frame and reports:
  - what fraction of frames the pen was confidently found in
  - an annotated preview video showing the detected pen blob + centroid
    + long-axis endpoints on every frame

Usage:
    python test_pen_detection.py --video uploads/sample.mp4 --paper A4
"""

import argparse
import os
import cv2
import numpy as np

from vision.video_processor import VideoProcessor
from vision.paper_detector import find_calibration_frame, draw_corners
from vision.pen_detector import build_paper_mask, detect_pen, draw_pen_debug

PAPER_SIZES_MM = {
    "A4": (210.0, 297.0),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--paper", default="A4")
    parser.add_argument("--width_mm", type=float, default=None)
    parser.add_argument("--height_mm", type=float, default=None)
    parser.add_argument("--out", default="debug_output")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.paper.upper() == "A4":
        width_mm, height_mm = PAPER_SIZES_MM["A4"]
    else:
        if args.width_mm is None or args.height_mm is None:
            raise SystemExit("Custom paper needs --width_mm and --height_mm")
        width_mm, height_mm = args.width_mm, args.height_mm

    print(f"Loading video: {args.video}")
    vp = VideoProcessor(args.video)
    for k, v in vp.summary().items():
        print(f"  {k}: {v}")

    print("\nLocating paper (Stage 1, reused)...")
    idx, calib_frame, corners = find_calibration_frame(vp)
    if corners is None:
        raise SystemExit("Could not find the paper. Run test_paper_detection.py first to debug.")
    print(f"Paper locked from frame {idx}.")

    paper_mask = build_paper_mask(calib_frame.shape, corners)

    print("\nScanning all frames for the pen...")
    writer = None
    found_count = 0
    total = 0
    sample_saved = {190: False, 240: False, 270: False}

    for i, frame in vp.frames(start=0, step=1):
        total += 1
        result = detect_pen(frame, paper_mask)
        if result["found"]:
            found_count += 1

        vis = draw_corners(frame, corners, color=(0, 200, 0))
        vis = draw_pen_debug(vis, result)
        cv2.putText(vis, f"frame {i}  pen_found={result['found']}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if writer is None:
            h, w = vis.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(os.path.join(args.out, "pen_detection_preview.mp4"),
                                      fourcc, vp.fps or 30.0, (w, h))
        writer.write(vis)

        if i in sample_saved and not sample_saved[i]:
            cv2.imwrite(os.path.join(args.out, f"pen_debug_frame_{i}.png"), vis)
            sample_saved[i] = True

    if writer is not None:
        writer.release()
    vp.release()

    rate = 100.0 * found_count / total if total else 0.0
    print(f"\nPen found in {found_count}/{total} frames ({rate:.1f}%).")
    print(f"Saved annotated preview -> {os.path.join(args.out, 'pen_detection_preview.mp4')}")
    print("Saved sample debug frames for 190, 240, 270 (if reached).")
    print("\nDone. Stage 2 (pen blob detection) complete.")


if __name__ == "__main__":
    main()
