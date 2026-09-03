"""
test_paper_detection.py

STAGE 1 MVP:
- Load a video
- Find the paper (auto-picks the clearest frame in the first few seconds)
- Perspective-correct it to a flat top-down rectangle at real-world scale
- Save proof images, and an annotated preview video with the paper
  outline drawn on every frame, so you can SEE the boundary tracking
  correctly for the whole clip.

USAGE:
    python test_paper_detection.py
    python test_paper_detection.py --video uploads/sample.mp4 --paper A4
    python test_paper_detection.py --video uploads/sample.mp4 --width-mm 210 --height-mm 297
"""

import argparse
import os
import cv2

from vision.video_processor import VideoProcessor
from vision.paper_detector import (
    find_calibration_frame,
    warp_paper,
    draw_corners,
)

OUTPUT_DIR = "debug_output"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="uploads/sample.mp4", help="Path to input video")
    parser.add_argument("--paper", choices=["A4", "custom"], default="A4")
    parser.add_argument("--width-mm", type=float, default=210.0, help="Used if --paper custom")
    parser.add_argument("--height-mm", type=float, default=297.0, help="Used if --paper custom")
    parser.add_argument("--px-per-mm", type=float, default=4.0)
    args = parser.parse_args()

    if args.paper == "A4":
        width_mm, height_mm = 210.0, 297.0
    else:
        width_mm, height_mm = args.width_mm, args.height_mm

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading video: {args.video}")
    vp = VideoProcessor(args.video)
    stats = vp.summary()
    print("Video info:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nSearching for the paper (scanning first ~3 seconds for the clearest view)...")
    cal_idx, cal_frame, corners = find_calibration_frame(vp, search_seconds=3.0)

    if corners is None:
        print("FAILED: Could not detect a rectangular paper in this video.")
        vp.release()
        return

    print(f"Found paper at frame {cal_idx} (t={cal_idx / vp.fps:.2f}s)")
    print("Corners (TL, TR, BR, BL) in pixels:")
    for label, (x, y) in zip(["TL", "TR", "BR", "BL"], corners):
        print(f"  {label}: ({x:.1f}, {y:.1f})")

    # Save the calibration frame with the detected boundary drawn on it.
    vis = draw_corners(cal_frame, corners)
    boundary_path = os.path.join(OUTPUT_DIR, "calibration_frame_boundary.png")
    cv2.imwrite(boundary_path, vis)
    print(f"\nSaved boundary overlay -> {boundary_path}")

    # Perspective-correct that frame to real-world scale.
    warped, matrix, px_per_mm = warp_paper(cal_frame, corners, width_mm, height_mm, args.px_per_mm)
    warped_path = os.path.join(OUTPUT_DIR, "warped_paper.png")
    cv2.imwrite(warped_path, warped)
    print(f"Saved perspective-corrected paper -> {warped_path}")
    print(f"Scale: {px_per_mm} px/mm  ->  warped size: {warped.shape[1]}x{warped.shape[0]} px "
          f"for a {width_mm:.0f}x{height_mm:.0f} mm sheet")

    # Since the camera is fixed for the whole video, reuse these same
    # corners on every frame and render an annotated preview video.
    # This is our visual proof that the boundary holds steady even once
    # the hand/pen start occluding part of the paper edge.
    print("\nRendering annotated preview video (this reads the whole clip)...")
    preview_path = os.path.join(OUTPUT_DIR, "annotated_preview.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(preview_path, fourcc, vp.fps or 30.0, (vp.width, vp.height))

    frame_written = 0
    for idx, frame in vp.frames(start=0, step=1):
        vis_frame = draw_corners(frame, corners)
        writer.write(vis_frame)
        frame_written += 1
    writer.release()
    print(f"Saved annotated preview ({frame_written} frames) -> {preview_path}")

    vp.release()
    print("\nDone. Stage 1 (video load + paper detection + perspective correction) complete.")


if __name__ == "__main__":
    main()
