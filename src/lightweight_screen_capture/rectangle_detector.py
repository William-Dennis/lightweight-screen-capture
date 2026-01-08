"""Simple rectangle detector using OpenCV for detecting rectangular regions."""

import cv2
import numpy as np
from typing import List, Tuple


def detect_rectangles(
    frame: np.ndarray,
    min_area: int = 10000,
    max_area: int = 500000,
) -> List[Tuple[int, int, int, int, float]]:
    """
    Detect rectangular regions in frame using edge detection.

    Args:
        frame: Input image (BGR format)
        min_area: Minimum rectangle area in pixels
        max_area: Maximum rectangle area in pixels

    Returns:
        List of (x1, y1, x2, y2, confidence) tuples
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    rectangles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not (min_area <= area <= max_area):
            continue

        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0

        if aspect_ratio < 0.1 or aspect_ratio > 10.0:
            continue

        contour_area = cv2.contourArea(contour)
        bbox_area = w * h
        fill_ratio = contour_area / bbox_area if bbox_area > 0 else 0

        confidence = min(1.0, fill_ratio)

        rectangles.append((x, y, x + w, y + h, confidence))

    rectangles = _remove_overlapping(rectangles)
    return rectangles


def _remove_overlapping(
    rectangles: List[Tuple[int, int, int, int, float]],
) -> List[Tuple[int, int, int, int, float]]:
    """Remove overlapping rectangles, keeping highest confidence."""
    if len(rectangles) == 0:
        return []

    rectangles = sorted(rectangles, key=lambda r: r[4], reverse=True)
    keep = []

    for rect in rectangles:
        overlaps = False
        for kept_rect in keep:
            if _calculate_iou(rect, kept_rect) > 0.5:
                overlaps = True
                break
        if not overlaps:
            keep.append(rect)

    return keep


def _calculate_iou(
    rect1: Tuple[int, int, int, int, float], rect2: Tuple[int, int, int, int, float]
) -> float:
    """Calculate Intersection over Union between two rectangles."""
    x1_1, y1_1, x2_1, y2_1 = rect1[:4]
    x1_2, y1_2, x2_2, y2_2 = rect2[:4]

    x1 = max(x1_1, x1_2)
    y1 = max(y1_1, y1_2)
    x2 = min(x2_1, x2_2)
    y2 = min(y2_1, y2_2)

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
