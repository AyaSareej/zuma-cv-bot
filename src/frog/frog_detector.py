from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import cv2
import numpy as np


@dataclass
class FrogDetection:
    center: Tuple[int, int]
    contour: np.ndarray
    score: float


def _circularity(area: float, perimeter: float) -> float:
    # circularity in [0..1+] (closer to 1 means more circle-like)
    if perimeter <= 1e-6:
        return 0.0
    return float((4.0 * np.pi * area) / (perimeter * perimeter))


def _solidity(cnt: np.ndarray) -> float:
    area = float(cv2.contourArea(cnt))
    hull = cv2.convexHull(cnt)
    hull_area = float(cv2.contourArea(hull))
    if hull_area <= 1e-6:
        return 0.0
    return area / hull_area


def detect_frog_by_contours(
    bgr: np.ndarray,
    hsv_lower: Tuple[int, int, int],
    hsv_upper: Tuple[int, int, int],
    min_area: int = 800,
    max_area: int = 200000,
) -> Optional[FrogDetection]:
    """
    Classical CV frog candidate detection:
    1) HSV threshold -> mask
    2) morphology -> clean
    3) findContours
    4) score candidates (area + circularity + solidity)
    Returns best candidate or None.

    IMPORTANT:
    - hsv ranges are game/theme dependent; tune them on your screenshot.
    - area thresholds depend on ROI size; tune them too.
    """

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower = np.array(hsv_lower, dtype=np.uint8)
    upper = np.array(hsv_upper, dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)

    # morphology: remove tiny noise and fill gaps
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    # contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best: Optional[FrogDetection] = None

    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area or area > max_area:
            continue

        peri = float(cv2.arcLength(cnt, True))
        circ = _circularity(area, peri)   # 0..1
        sol = float(_solidity(cnt))       # 0..1

        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / max(1.0, float(h))
        # Prefer roughly "compact" shapes
        aspect_score = 1.0 - min(1.0, abs(aspect - 1.0))  # best near 1

        # Simple score: tune weights later
        score = (0.45 * circ) + (0.35 * sol) + (0.20 * aspect_score)

        # center from moments
        M = cv2.moments(cnt)
        if abs(M.get("m00", 0.0)) < 1e-6:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        det = FrogDetection(center=(cx, cy), contour=cnt, score=float(score))
        if best is None or det.score > best.score:
            best = det

    return best


def draw_frog_debug(
    bgr: np.ndarray,
    detection: Optional[FrogDetection],
    all_contours: Optional[List[np.ndarray]] = None
) -> np.ndarray:
    out = bgr.copy()
    if all_contours is not None and len(all_contours) > 0:
        cv2.drawContours(out, all_contours, -1, (0, 255, 255), 2)

    if detection is not None:
        cv2.drawContours(out, [detection.contour], -1, (0, 255, 0), 3)
        cx, cy = detection.center
        cv2.circle(out, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(out, f"frog_center=({cx},{cy}) score={detection.score:.3f}",
                    (max(0, cx - 140), max(20, cy - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out
