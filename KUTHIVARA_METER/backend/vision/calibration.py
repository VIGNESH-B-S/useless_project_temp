"""Lock the paper geometry once, then reuse it for an entire fixed-camera video."""

from __future__ import annotations

from dataclasses import dataclass
import json

import cv2
import numpy as np

from vision.paper_detector import find_calibration_frame, warp_paper


PAPER_SIZES_MM = {"A4": (210.0, 297.0)}


@dataclass
class PaperCalibration:
    """The fixed mapping between camera pixels and real paper millimetres."""

    frame_index: int
    corners: np.ndarray
    homography: np.ndarray
    width_mm: float
    height_mm: float
    px_per_mm: float

    @property
    def output_size_px(self) -> tuple[int, int]:
        return (round(self.width_mm * self.px_per_mm), round(self.height_mm * self.px_per_mm))

    def warp(self, frame: np.ndarray) -> np.ndarray:
        """Perspective-correct one frame into top-down paper coordinates."""
        return cv2.warpPerspective(frame, self.homography, self.output_size_px)

    def image_point_to_mm(self, point: tuple[float, float]) -> tuple[float, float]:
        """Convert one camera-image point to paper coordinates in millimetres."""
        source = np.asarray([[point]], dtype=np.float32)
        warped = cv2.perspectiveTransform(source, self.homography)[0, 0]
        return float(warped[0] / self.px_per_mm), float(warped[1] / self.px_per_mm)

    def save_json(self, path: str) -> None:
        """Save calibration values for inspection or reuse in a later run."""
        data = {
            "frame_index": self.frame_index,
            "corners": self.corners.tolist(),
            "homography": self.homography.tolist(),
            "width_mm": self.width_mm,
            "height_mm": self.height_mm,
            "px_per_mm": self.px_per_mm,
            "output_size_px": list(self.output_size_px),
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)


def calibrate_video(video_processor, paper: str = "A4", px_per_mm: float = 4.0,
                    search_seconds: float = 3.0, width_mm: float | None = None,
                    height_mm: float | None = None) -> PaperCalibration:
    """Find the cleanest paper frame and return its locked calibration.

    The camera must remain fixed after this point.  If it moves, calibrate
    again rather than reusing the old homography.
    """
    if paper.upper() in PAPER_SIZES_MM:
        width_mm, height_mm = PAPER_SIZES_MM[paper.upper()]
    elif width_mm is None or height_mm is None or width_mm <= 0 or height_mm <= 0:
        raise ValueError("Custom paper requires positive width_mm and height_mm values.")
    frame_index, frame, corners = find_calibration_frame(video_processor, search_seconds)
    if corners is None:
        raise RuntimeError("Could not find the paper for calibration.")

    _, homography, _ = warp_paper(frame, corners, width_mm, height_mm, px_per_mm)
    return PaperCalibration(
        frame_index=frame_index,
        corners=corners,
        homography=homography,
        width_mm=width_mm,
        height_mm=height_mm,
        px_per_mm=px_per_mm,
    )
