"""
pen_detector.py

Finds the pen in a single video frame.

HOW IT WORKS (in plain terms)
------------------------------
1. We only look for the pen INSIDE the paper region (using the locked
   paper corners from paper_detector.py). This throws away a huge source
   of false positives: dark keyboard keys, monitor reflections, etc. all
   sit outside the paper, so restricting the search area removes them
   for free.
2. Within that region, threshold on HSV for "pen blue". We picked this
   range by sampling actual pen pixels from the sample video (median
   H=115, S=118, V=103), not by guessing.
3. Clean up the mask with morphology (removes speckle noise).
4. Take the largest surviving contour as "the pen". If it's too small,
   we say "no pen detected this frame" rather than guessing.
5. Report both a simple centroid AND a fitted line (via minAreaRect),
   since the visible blue sliver is usually just part of the pen barrel
   between the fingers -- the *long axis* of that sliver points along
   the pen, which we'll use next stage to estimate the actual tip.

NOTE: the pen will not be visible in every frame (hand may fully cover
it, or it may be out of shot). That's expected and handled later by
the trajectory tracker via interpolation -- this module just honestly
reports "found" or "not found" per frame.
"""

import cv2
import numpy as np

# Tuned from real pixels sampled off the sample pen (a blue ballpoint
# with a clear/white grip section).
DEFAULT_HSV_LOWER = (95, 40, 40)
DEFAULT_HSV_UPPER = (135, 255, 255)

# Minimum blob area (in pixels) to count as a real pen detection,
# not a stray speck of noise.
MIN_PEN_AREA = 40


def build_paper_mask(frame_shape, corners: np.ndarray, dilate_px: int = 6) -> np.ndarray:
    """
    Build a binary mask (255 inside, 0 outside) covering the paper region,
    slightly dilated so we don't clip the pen tip right at the paper edge.
    """
    h, w = frame_shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = corners.astype(np.int32).reshape(1, 4, 2)
    cv2.fillPoly(mask, pts, 255)

    if dilate_px > 0:
        kernel = np.ones((dilate_px, dilate_px), np.uint8)
        mask = cv2.dilate(mask, kernel)

    return mask


def detect_pen(frame: np.ndarray,
                paper_mask: np.ndarray,
                hsv_lower=DEFAULT_HSV_LOWER,
                hsv_upper=DEFAULT_HSV_UPPER,
                min_area: float = MIN_PEN_AREA) -> dict:
    """
    Look for the pen in one frame, restricted to `paper_mask`.

    Returns a dict:
        {
            "found": bool,
            "centroid": (x, y) or None,
            "contour": contour or None,
            "area": float,
            "endpoints": (ptA, ptB) or None,  # long axis of the blob,
                                               # two tip *candidates*
        }
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
    mask = cv2.bitwise_and(mask, paper_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"found": False, "centroid": None, "contour": None, "area": 0, "endpoints": None}

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)

    if area < min_area:
        return {"found": False, "centroid": None, "contour": None, "area": area, "endpoints": None}

    m = cv2.moments(largest)
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]

    # Fit the long axis of the blob so we know which way the pen points.
    rect = cv2.minAreaRect(largest)  # ((cx,cy),(w,h),angle)
    box = cv2.boxPoints(rect)  # 4 corners of the rotated rect
    # The long axis endpoints = midpoints of the two SHORT sides.
    d01 = np.linalg.norm(box[0] - box[1])
    d12 = np.linalg.norm(box[1] - box[2])
    if d01 < d12:
        ptA = (box[0] + box[1]) / 2.0
        ptB = (box[2] + box[3]) / 2.0
    else:
        ptA = (box[1] + box[2]) / 2.0
        ptB = (box[3] + box[0]) / 2.0

    return {
        "found": True,
        "centroid": (float(cx), float(cy)),
        "contour": largest,
        "area": float(area),
        "endpoints": (tuple(ptA), tuple(ptB)),
    }


def draw_pen_debug(frame: np.ndarray, result: dict) -> np.ndarray:
    """Return a copy of frame with the pen detection drawn on it."""
    vis = frame.copy()
    if not result["found"]:
        return vis

    cv2.drawContours(vis, [result["contour"]], -1, (0, 255, 255), 2)
    cx, cy = result["centroid"]
    cv2.circle(vis, (int(cx), int(cy)), 5, (0, 0, 255), -1)

    if result["endpoints"] is not None:
        ptA, ptB = result["endpoints"]
        cv2.circle(vis, (int(ptA[0]), int(ptA[1])), 5, (255, 0, 255), -1)
        cv2.circle(vis, (int(ptB[0]), int(ptB[1])), 5, (255, 128, 0), -1)
        cv2.line(vis, (int(ptA[0]), int(ptA[1])), (int(ptB[0]), int(ptB[1])), (255, 0, 255), 1)

    return vis
