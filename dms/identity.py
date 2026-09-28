"""Which driver is which.

Each camera shot gets an appearance signature for its driver: a colour histogram
of the upper body (shoulders to hips), averaged over the shot. Shots whose
signatures are close are the same person. A new driver means a clearly
different signature, e.g. a red jacket instead of a beige one.

This is appearance, not identity: it can't tell apart two people dressed alike,
and the same person changing jackets would look like a new driver. Face
recognition would be the next step for a real fleet system.
"""

from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from .perception import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER


def upper_body_signature(frame: np.ndarray, person: dict) -> list[float] | None:
    x1, y1, x2, y2 = (int(v) for v in person["box"])
    k = person.get("kpts")
    if k is not None:
        pts = [k[i] for i in (L_SHOULDER, R_SHOULDER, L_HIP, R_HIP) if k[i][2] > 0.3]
        if len(pts) >= 2:
            xs, ys = [p[0] for p in pts], [p[1] for p in pts]
            pad = 0.15 * (max(xs) - min(xs) + 1)
            x1, x2 = int(min(xs) - pad), int(max(xs) + pad)
            y1 = int(min(ys))
            y2 = int(max(max(ys), min(ys) + 0.5 * (y2 - y1)))
    h, w = frame.shape[:2]
    x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
    if x2 - x1 < 10 or y2 - y1 < 10:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
    return [round(float(v), 5) for v in hist]


def assign_drivers(shots: list[int], signatures: list[list[float] | None], threshold: float = 0.45) -> dict[int, int]:
    """Map shot index -> driver number (1, 2, ...), in order of first appearance."""
    per_shot = defaultdict(list)
    for s, sig in zip(shots, signatures, strict=True):
        if sig is not None:
            per_shot[s].append(np.asarray(sig, np.float32))
    means = {s: np.mean(v, axis=0) for s, v in per_shot.items()}

    drivers: list[np.ndarray] = []   # running mean signature per driver
    counts: list[int] = []
    mapping: dict[int, int] = {}
    last = None
    for s in sorted(set(shots)):
        if s not in means:
            mapping[s] = last if last is not None else 1
            continue
        sig = means[s]
        dists = [cv2.compareHist(d, sig, cv2.HISTCMP_BHATTACHARYYA) for d in drivers]
        if dists and min(dists) < threshold:
            idx = int(np.argmin(dists))
            drivers[idx] = (drivers[idx] * counts[idx] + sig) / (counts[idx] + 1)
            counts[idx] += 1
        else:
            drivers.append(sig)
            counts.append(1)
            idx = len(drivers) - 1
        mapping[s] = idx + 1
        last = idx + 1
    return mapping
