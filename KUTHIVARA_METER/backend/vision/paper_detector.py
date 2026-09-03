"""
paper_detector.py

Finds the white sheet of paper in a video frame and gives us:
1. Its 4 corners (ordered TL, TR, BR, BL).
2. A "bird's eye view" (perspective-corrected) crop of just the paper,
   scaled to match real-world mm dimensions.

HOW IT WORKS (in plain terms)
------------------------------
1. Convert the frame to HSV instead of plain RGB. HSV separates
   "how colorful" (saturation) from "how bright" (value). White paper
   is bright AND low-saturation, which makes it easy to isolate even
   next to warm-colored reflections (like the orange light in our
   sample video) that are bright but ARE colorful (high saturation).
2. Threshold on that mask to get a black/white image of "paper vs not paper".
3. Clean up the mask with morphology (removes small speckles, fills tiny holes).
4. Find the largest connected white blob (contour).
5. Take its CONVEX HULL. This is the key trick: when a hand is resting on
   part of the paper's edge, the raw contour gets a "bite" taken out of it
   (no longer a clean rectangle). The convex hull "shrink-wraps" around the
   hand intrusion and gives us back a clean quadrilateral.
6. Approximate that hull as a polygon. If it simplifies to ~4 points, we've
   found our paper corners.
"""

import cv2
import numpy as np

# HSV range for "white paper": low saturation, high brightness.
# Works regardless of hue, so it isn't fooled by warm lighting tint.
DEFAULT_HSV_LOWER = (0, 0, 150)
DEFAULT_HSV_UPPER = (180, 60, 255)

# Paper must occupy at least this fraction of the frame area to count.
MIN_AREA_RATIO = 0.05


def _order_points(pts: np.ndarray) -> np.ndarray:
    """
    Given 4 (x, y) points in any order, return them ordered as:
    top-left, top-right, bottom-right, bottom-left.
    Standard trick: TL has smallest (x+y), BR has largest (x+y),
    TR has smallest (y-x), BL has largest (y-x).
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right

    diff = pts[:, 1] - pts[:, 0]  # y - x
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left

    return ordered


def detect_paper_corners(frame: np.ndarray,
                          hsv_lower=DEFAULT_HSV_LOWER,
                          hsv_upper=DEFAULT_HSV_UPPER,
                          min_area_ratio: float = MIN_AREA_RATIO):
    """
    Try to find the paper in this single frame.

    Returns:
        (corners, area) where corners is a (4,2) float32 array ordered
        TL, TR, BR, BL, and area is the detected paper area in pixels.
        Returns (None, 0) if no confident quadrilateral was found.
    """
    h, w = frame.shape[:2]
    frame_area = h * w

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area_ratio * frame_area:
        return None, 0

    # Convex hull fixes the "bite taken out by hand" problem.
    hull = cv2.convexHull(largest)

    # Try a few epsilon values until we get a clean quadrilateral.
    peri = cv2.arcLength(hull, True)
    approx = None
    for eps_frac in (0.02, 0.01, 0.03, 0.04, 0.05):
        candidate = cv2.approxPolyDP(hull, eps_frac * peri, True)
        if len(candidate) == 4:
            approx = candidate
            break

    if approx is None:
        return None, 0

    corners = _order_points(approx)
    return corners, area


def find_calibration_frame(video_processor, search_seconds: float = 3.0):
    """
    Scan the first `search_seconds` of the video (paper is usually still
    clean/unoccluded right at the start) and pick the frame that gives the
    most confident (largest-area) paper detection. Since the camera is
    fixed for the whole recording, this one set of corners can be reused
    for every subsequent frame.

    Returns:
        (frame_index, frame, corners) or (None, None, None) if nothing found.
    """
    fps = video_processor.fps or 30.0
    search_frames = int(fps * search_seconds)
    search_frames = min(search_frames, video_processor.frame_count)

    best = None  # (area, index, frame, corners)

    for idx, frame in video_processor.frames(start=0, step=1):
        if idx >= search_frames:
            break
        corners, area = detect_paper_corners(frame)
        if corners is not None:
            if best is None or area > best[0]:
                best = (area, idx, frame.copy(), corners)

    # Fallback: if nothing found in the initial window, scan the whole video.
    if best is None:
        for idx, frame in video_processor.frames(start=0, step=max(1, int(fps // 5))):
            corners, area = detect_paper_corners(frame)
            if corners is not None:
                if best is None or area > best[0]:
                    best = (area, idx, frame.copy(), corners)

    if best is None:
        return None, None, None

    _, idx, frame, corners = best
    return idx, frame, corners


def warp_paper(frame: np.ndarray, corners: np.ndarray,
                width_mm: float, height_mm: float, px_per_mm: float = 4.0):
    """
    Perspective-correct the frame so the paper region becomes a flat,
    top-down rectangle sized to match its real physical dimensions
    (at `px_per_mm` pixels per millimetre). This is what establishes our
    pixel-to-real-world scale for every later measurement.

    Returns:
        (warped_image, transform_matrix, px_per_mm)
    """
    out_w = max(1, int(round(width_mm * px_per_mm)))
    out_h = max(1, int(round(height_mm * px_per_mm)))

    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1],
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, matrix, (out_w, out_h))

    return warped, matrix, px_per_mm


def draw_corners(frame: np.ndarray, corners: np.ndarray, color=(0, 0, 255)) -> np.ndarray:
    """Return a copy of frame with the paper outline + corner dots drawn on it."""
    vis = frame.copy()
    pts = corners.astype(int)
    cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=3)
    labels = ["TL", "TR", "BR", "BL"]
    for p, label in zip(pts, labels):
        cv2.circle(vis, tuple(p), 6, (0, 255, 0), -1)
        cv2.putText(vis, label, (p[0] + 8, p[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return vis
