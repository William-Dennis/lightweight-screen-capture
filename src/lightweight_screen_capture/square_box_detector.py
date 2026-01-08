"""
Improved Rectangle Detector - Detects main content images + UI elements
Maintains original API: detect_square_regions() -> List[Tuple[int, int, int, int, float, int]]
"""

import cv2
import numpy as np
from typing import List, Tuple


class SquareBoxDetector:
    """Detects rectangular regions including main images, ads, and UI elements."""

    def __init__(
        self,
        min_area: int = 10000,
        max_area: int = 2000000,  # Increased to catch large images
        aspect_ratio_tolerance: float = 0.3,
    ):
        """
        Initialize rectangle detector.
        
        Args:
            min_area: Minimum rectangle area in pixels
            max_area: Maximum rectangle area in pixels (increased for main images)
            aspect_ratio_tolerance: Tolerance for aspect ratio (unused now)
        """
        self.min_area = min_area
        self.max_area = max_area
        self.aspect_ratio_tolerance = aspect_ratio_tolerance

    def detect_square_regions(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Detect rectangular regions in frame using multiple strategies.
        
        Args:
            frame: Input image (BGR format)
            
        Returns:
            List of (x1, y1, x2, y2, confidence, class_id) tuples
        """
        regions = []
        
        # Strategy 1: Edge-based detection (good for buttons, borders)
        regions.extend(self._detect_by_edges(frame))
        
        # Strategy 2: Color contrast detection (good for main images)
        regions.extend(self._detect_by_contrast(frame))
        
        # Remove duplicates
        regions = self._remove_duplicates(regions)
        
        return regions
    
    def _detect_by_edges(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """Detect rectangles using edge detection (original method)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area <= area <= self.max_area):
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
            conf = min(1.0, fill_ratio) * 0.7  # Lower confidence for edge-based
            
            regions.append((x, y, x + w, y + h, conf, 0))
        
        return regions
    
    def _detect_by_contrast(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float, int]]:
        """
        Detect rectangles by finding areas with high contrast to background.
        Good for finding main images on dark/light backgrounds.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to find bright/dark regions
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Also try inverted
        binary_inv = cv2.bitwise_not(binary)
        
        regions = []
        
        for binary_img in [binary, binary_inv]:
            # Clean up the binary image
            kernel = np.ones((5, 5), np.uint8)
            binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=3)
            binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if not (self.min_area <= area <= self.max_area):
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if it's reasonably rectangular
                contour_area = cv2.contourArea(contour)
                bbox_area = w * h
                if bbox_area == 0:
                    continue
                    
                fill_ratio = contour_area / bbox_area
                
                # Must fill at least 70% of bounding box to be rectangular
                if fill_ratio < 0.7:
                    continue
                
                # Calculate confidence based on size and fill ratio
                # Larger regions get higher confidence (likely main content)
                size_factor = min(1.0, area / (self.max_area * 0.5))
                conf = (fill_ratio * 0.7 + size_factor * 0.3)
                
                regions.append((x, y, x + w, y + h, conf, 0))
        
        return regions
    
    def _remove_duplicates(
        self, regions: List[Tuple[int, int, int, int, float, int]]
    ) -> List[Tuple[int, int, int, int, float, int]]:
        """Remove overlapping detections, keeping the highest confidence."""
        if len(regions) == 0:
            return []
        
        # Sort by confidence (descending)
        regions = sorted(regions, key=lambda r: r[4], reverse=True)
        
        keep = []
        
        for region in regions:
            # Check if this region overlaps significantly with any kept region
            overlaps = False
            for kept_region in keep:
                if self._calculate_iou(region, kept_region) > 0.5:
                    overlaps = True
                    break
            
            if not overlaps:
                keep.append(region)
        
        return keep
    
    def _calculate_iou(
        self,
        region1: Tuple[int, int, int, int, float, int],
        region2: Tuple[int, int, int, int, float, int]
    ) -> float:
        """Calculate Intersection over Union between two regions."""
        x1_1, y1_1, x2_1, y2_1 = region1[:4]
        x1_2, y1_2, x2_2, y2_2 = region2[:4]
        
        # Calculate intersection
        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0