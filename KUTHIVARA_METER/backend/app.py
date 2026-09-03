"""KUTHIVARA Meter Flask application.

Run from this directory with: ``python app.py``
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

from vision.calibration import calibrate_video
from vision.ink_estimator import estimate_ink
from vision.paper_detector import draw_corners
from vision.pen_detector import build_paper_mask, detect_pen
from vision.trajectory_tracker import TrajectoryTracker
from vision.video_processor import VideoProcessor


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR.parent / "frontend"
UPLOAD_DIR = BASE_DIR / "uploads"
RESULTS_DIR = BASE_DIR / "static" / "results"
ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static")
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response
app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # 300 MB


def _allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _number(value: str | None, name: str, minimum: float = 0.0) -> float:
    try:
        parsed = float(value or "")
    except ValueError as error:
        raise ValueError(f"{name} must be a number.") from error
    if parsed <= minimum:
        raise ValueError(f"{name} must be greater than {minimum}.")
    return parsed


def _draw_trajectory(frame: np.ndarray, tracker: TrajectoryTracker) -> np.ndarray:
    """Render only actual accepted motion segments, never an invented long gap."""
    view = frame.copy()
    for start, end in tracker.segments:
        cv2.line(view, tuple(start.point_px.astype(int)), tuple(end.point_px.astype(int)),
                 (0, 230, 255), 2, cv2.LINE_AA)
    if tracker.points:
        cv2.circle(view, tuple(tracker.points[-1].point_px.astype(int)), 5, (30, 40, 255), -1)
    return view


def analyze_video(video_path: Path, result_dir: Path, paper_type: str, width_mm: float,
                  height_mm: float, ink_rate_ml_per_m: float) -> dict:
    """Run the existing OpenCV pipeline and return website-ready JSON values."""
    started = time.perf_counter()
    vp = VideoProcessor(str(video_path))
    try:
        if vp.frame_count < 30:
            raise ValueError("Video is too short. Please upload at least one second of footage.")

        calibration = calibrate_video(
            vp, paper=paper_type, width_mm=width_mm, height_mm=height_mm, px_per_mm=4.0
        )
        calibration_frame = vp.get_frame(calibration.frame_index)
        if calibration_frame is None:
            raise RuntimeError("Could not read the calibration frame.")

        paper_mask = build_paper_mask(calibration_frame.shape, calibration.corners)
        trajectory_path = result_dir / "trajectory.png"
        boundary_path = result_dir / "paper-boundary.png"
        calibration.save_json(str(result_dir / "calibration.json"))
        cv2.imwrite(str(boundary_path), draw_corners(calibration_frame, calibration.corners))

        writer = cv2.VideoWriter(
            str(result_dir / "trajectory-preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
            vp.fps or 30.0, calibration.output_size_px,
        )
        tracker = TrajectoryTracker(calibration.homography, calibration.px_per_mm)
        final_view = None
        pen_found = 0
        for frame_index, frame in vp.frames():
            pen = detect_pen(frame, paper_mask)
            if pen["found"]:
                pen_found += 1
            tracker.update(frame_index, pen)
            warped = calibration.warp(frame)
            final_view = _draw_trajectory(warped, tracker)
            writer.write(final_view)
        writer.release()

        if final_view is None or not tracker.segments:
            raise ValueError("No usable pen trajectory was found. Ensure the pen is visible over the paper.")
        cv2.imwrite(str(trajectory_path), final_view)

        estimate = estimate_ink(tracker.length_mm, ink_rate_ml_per_m / 100.0)
        track_frames = [point.frame_index for point in tracker.points]
        active_span = max(1, track_frames[-1] - track_frames[0] + 1)
        active_coverage = tracker.accepted_frames / active_span
        processing_seconds = time.perf_counter() - started
        return {
            "scribble_length_mm": round(tracker.length_mm, 2),
            "scribble_length_cm": round(tracker.length_mm / 10.0, 2),
            "scribble_length_m": round(tracker.length_mm / 1000.0, 3),
            "estimated_ink_ml": estimate.estimated_ink_ml,
            "estimated_ink_microlitres": estimate.estimated_ink_microlitres,
            "ink_rate_ml_per_m": ink_rate_ml_per_m,
            "processing_seconds": round(processing_seconds, 2),
            "confidence": round(min(1.0, active_coverage), 3),
            "tracking": {
                "video_frames": vp.frame_count,
                "pen_detected_frames": pen_found,
                "accepted_track_points": tracker.accepted_frames,
                "active_window_coverage": round(active_coverage, 3),
            },
        }
    finally:
        vp.release()


@app.get("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.get("/health")
def health():
    """Simple deployment check that does not depend on frontend files."""
    return jsonify(status="ok", service="KUTHIVARA Meter API")


@app.get("/frontend/<path:filename>")
def frontend_file(filename: str):
    return send_from_directory(FRONTEND_DIR, filename)


@app.post("/api/analyze")
def analyze():
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify(success=False, error="Choose a video before analysing."), 400
    if not _allowed_file(video.filename):
        return jsonify(success=False, error="Use an MP4, MOV, AVI, or MKV video."), 400

    try:
        paper_type = request.form.get("paper_type", "A4").upper()
        if paper_type == "A4":
            width_mm, height_mm = 210.0, 297.0
        elif paper_type == "CUSTOM":
            width_mm = _number(request.form.get("width_mm"), "Paper width")
            height_mm = _number(request.form.get("height_mm"), "Paper height")
        else:
            raise ValueError("Choose A4 or Custom paper.")
        ink_rate_ml_per_m = _number(request.form.get("ink_rate_ml_per_m"), "Ink rate")
    except ValueError as error:
        return jsonify(success=False, error=str(error)), 400

    job_id = uuid.uuid4().hex
    filename = f"{job_id}_{secure_filename(video.filename)}"
    video_path = UPLOAD_DIR / filename
    result_dir = RESULTS_DIR / job_id
    result_dir.mkdir()
    video.save(video_path)

    try:
        report = analyze_video(video_path, result_dir, paper_type, width_mm, height_mm, ink_rate_ml_per_m)
    except Exception as error:
        return jsonify(success=False, error=f"Analysis could not finish: {error}"), 422

    pen = {
        "brand": request.form.get("pen_brand", "Not specified").strip() or "Not specified",
        "model": request.form.get("pen_model", "Not specified").strip() or "Not specified",
        "tip_size_mm": request.form.get("tip_size_mm", "Not specified").strip() or "Not specified",
        "color": request.form.get("pen_color", "Not specified").strip() or "Not specified",
        "calibration": "User-provided fixed deposition rate",
    }
    report.update({
        "success": True,
        "paper": {"type": paper_type, "width_mm": width_mm, "height_mm": height_mm},
        "pen": pen,
        "trajectory_image_url": f"/static/results/{job_id}/trajectory.png",
        "paper_boundary_url": f"/static/results/{job_id}/paper-boundary.png",
        "trajectory_video_url": f"/static/results/{job_id}/trajectory-preview.mp4",
        "note": "Estimated Ink Volume uses your supplied fixed pen-deposition rate; it is not a direct camera measurement of liquid ink.",
    })
    return jsonify(report)


@app.errorhandler(413)
def too_large(_error):
    return jsonify(success=False, error="Video is too large. The current limit is 300 MB."), 413


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
