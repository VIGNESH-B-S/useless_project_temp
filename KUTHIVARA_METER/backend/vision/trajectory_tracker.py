"""A small, conservative pen-trajectory tracker.

This module deliberately tracks a *pen-end estimate*, not a claimed ink
contact point.  The current ``pen_detector`` only sees a blue portion of the
barrel, which can be displaced from the real tip by the hand.  Consequently,
``length_mm`` is a baseline diagnostic until a later contact-point stage is
added.

Every accepted point is transformed by the paper homography before distances
are measured, so the reported units are not perspective-distorted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


def transform_point(point: tuple[float, float], homography: np.ndarray) -> np.ndarray:
    """Map one image-space point to perspective-corrected paper pixels."""
    source = np.asarray([[point]], dtype=np.float32)
    return cv2.perspectiveTransform(source, homography)[0, 0]


@dataclass
class TrackPoint:
    frame_index: int
    point_px: np.ndarray


class TrajectoryTracker:
    """Choose a stable pen endpoint and accumulate short-gap path segments.

    The first detection chooses the endpoint lowest on the flattened page.
    Later detections choose whichever endpoint is nearest the previous one.
    A large motion or a long invisible gap starts a new track: no fictional
    straight line is added across it.
    """

    def __init__(
        self,
        homography: np.ndarray,
        px_per_mm: float,
        max_gap_frames: int = 3,
        max_step_mm: float = 15.0,
    ) -> None:
        if px_per_mm <= 0:
            raise ValueError("px_per_mm must be positive")
        self.homography = homography.astype(np.float32)
        self.px_per_mm = float(px_per_mm)
        self.max_gap_frames = int(max_gap_frames)
        self.max_step_px = float(max_step_mm) * self.px_per_mm

        self.points: list[TrackPoint] = []
        self.segments: list[tuple[TrackPoint, TrackPoint]] = []
        self.length_px = 0.0
        self.detected_frames = 0
        self.accepted_frames = 0
        self.total_frames = 0
        self._last: Optional[TrackPoint] = None

    def update(self, frame_index: int, pen_result: dict) -> Optional[np.ndarray]:
        """Consume one ``detect_pen`` result and return its accepted page point.

        Returns ``None`` for an absent pen, a long gap, or an implausibly large
        one-frame jump.  Such frames never add length.
        """
        self.total_frames += 1
        if not pen_result.get("found") or pen_result.get("endpoints") is None:
            return None

        self.detected_frames += 1
        endpoint_a, endpoint_b = pen_result["endpoints"]
        candidates = [
            transform_point(endpoint_a, self.homography),
            transform_point(endpoint_b, self.homography),
        ]

        if self._last is None or frame_index - self._last.frame_index > self.max_gap_frames:
            # Page y increases downwards.  This makes the initial choice
            # deterministic; subsequent observations use continuity instead.
            chosen = max(candidates, key=lambda point: float(point[1]))
            self._last = TrackPoint(frame_index, chosen)
            self.points.append(self._last)
            self.accepted_frames += 1
            return chosen.copy()

        chosen = min(candidates, key=lambda point: float(np.linalg.norm(point - self._last.point_px)))
        step_px = float(np.linalg.norm(chosen - self._last.point_px))
        if step_px > self.max_step_px:
            # Keep the existing point.  A false blob must not create a long
            # distance spike or pull the track to another part of the page.
            return None

        current = TrackPoint(frame_index, chosen)
        self.length_px += step_px
        self.segments.append((self._last, current))
        self.points.append(current)
        self._last = current
        self.accepted_frames += 1
        return chosen.copy()

    @property
    def length_mm(self) -> float:
        return self.length_px / self.px_per_mm

    def summary(self) -> dict:
        coverage = self.detected_frames / self.total_frames if self.total_frames else 0.0
        return {
            "frames": self.total_frames,
            "pen_detected_frames": self.detected_frames,
            "accepted_track_points": self.accepted_frames,
            "detection_coverage": coverage,
            "segments": len(self.segments),
            "length_mm": self.length_mm,
        }
