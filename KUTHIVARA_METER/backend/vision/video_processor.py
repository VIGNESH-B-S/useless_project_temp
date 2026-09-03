"""
video_processor.py

Small wrapper around OpenCV's VideoCapture.
Handles opening a video file and gives us easy access to:
- fps, width, height, frame_count, duration
- reading a specific frame by index
- iterating over frames (optionally skipping some, for speed)
"""

import cv2


class VideoProcessor:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise IOError(f"Could not open video file: {path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.frame_count / self.fps if self.fps > 0 else 0.0

    def get_frame(self, index: int):
        """Random-access read of a single frame by index. Returns None if out of range."""
        if index < 0 or index >= self.frame_count:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        return frame if ok else None

    def frames(self, start: int = 0, step: int = 1):
        """
        Generator that yields (frame_index, frame) from `start` to the end,
        advancing `step` frames at a time. step=1 means every frame.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        idx = start
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, frame
            idx += step
            if step > 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

    def summary(self) -> dict:
        return {
            "path": self.path,
            "fps": round(self.fps, 3),
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "duration_sec": round(self.duration_sec, 3),
        }

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
