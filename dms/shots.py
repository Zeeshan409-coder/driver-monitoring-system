"""Camera cut detection.

The behaviour logic calibrates itself per camera shot (where the driver normally
sits, where their hands normally are), so it needs to know when the camera or
car changes. A cut is a big jump in the colour histogram between consecutive
analysed frames.
"""

from __future__ import annotations

import cv2
import numpy as np


class ShotDetector:
    def __init__(self, threshold: float = 0.35, min_shot_s: float = 1.0):
        self.threshold = threshold
        self.min_shot_s = min_shot_s
        self.prev = None
        self.shot = 0
        self.shot_start = 0.0

    @staticmethod
    def signature(frame: np.ndarray) -> np.ndarray:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        hist = cv2.calcHist([cv2.cvtColor(small, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 16], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()

    def update(self, frame: np.ndarray, t: float) -> tuple[int, float]:
        """Returns (shot index, distance to previous frame)."""
        sig = self.signature(frame)
        dist = 0.0
        if self.prev is not None:
            dist = float(cv2.compareHist(self.prev, sig, cv2.HISTCMP_BHATTACHARYYA))
            if dist > self.threshold and t - self.shot_start >= self.min_shot_s:
                self.shot += 1
                self.shot_start = t
        self.prev = sig
        return self.shot, dist
