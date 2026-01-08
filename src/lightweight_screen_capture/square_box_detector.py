"""Simple square region detector using OpenCV for low-latency detection."""

import cv2
import numpy as np
from typing import List, Tuple


class SquareBoxDetector:
    """Detects square/rectangular regions like ads and image galleries using cv2."""

    def __init__(
        self,
        min_area: int = 10000,
        max_area: int = 500000,
        aspect_ratio_tolerance: float = 0.3,
    ):
        """Initialize square box detector with area and aspect ratio constraints."""
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_tolerance = aspect_ratio_tolerance

    def detect_square_regions(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float, int]]:
        """Detect square regions in frame, returns list of (x1, y1, x2, y2, conf, cls)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area <= area <= self.max_area:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if abs(aspect_ratio - 1.0) <= self.aspect_ratio_tolerance:
                    conf = min(1.0, area / self.max_area)
                    regions.append((x, y, x + w, y + h, conf, 0))
        return regions
