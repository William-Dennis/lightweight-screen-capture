"""Temporal tracking and smoothing for detected regions."""

import numpy as np
from typing import List, Tuple, Dict
from collections import deque


class RegionTracker:
    """Tracks regions over time and smooths detection scores."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        history_size: int = 5,
        score_threshold: float = 0.6,
    ):
        """Initialize region tracker with IOU and history parameters."""
        self.iou_threshold = iou_threshold
        self.history_size = history_size
        self.score_threshold = score_threshold
        self.tracked_regions: Dict[int, dict] = {}
        self.next_id = 0

    def _compute_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Compute IOU between two boxes (x1, y1, x2, y2)."""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        inter_area = max(0, x2_min - x1_max) * max(0, y2_min - y1_max)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update(self, detections: List[Tuple]) -> List[Tuple[int, Tuple, float]]:
        """Update tracked regions and return (track_id, box, smooth_score)."""
        matched_ids = set()
        updated_tracks = []
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            box = (x1, y1, x2, y2)
            best_iou, best_id = 0, None
            for track_id, track in self.tracked_regions.items():
                iou = self._compute_iou(box, track["box"])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou, best_id = iou, track_id
            if best_id is not None:
                track = self.tracked_regions[best_id]
                track["scores"].append(conf)
                track["box"] = box
                track["cls"] = cls
                matched_ids.add(best_id)
            else:
                best_id = self.next_id
                self.next_id += 1
                self.tracked_regions[best_id] = {
                    "box": box,
                    "scores": deque([conf], maxlen=self.history_size),
                    "cls": cls,
                }
                matched_ids.add(best_id)
        self.tracked_regions = {
            k: v for k, v in self.tracked_regions.items() if k in matched_ids
        }
        for track_id, track in self.tracked_regions.items():
            smooth_score = np.mean(track["scores"])
            if smooth_score >= self.score_threshold:
                updated_tracks.append((track_id, track["box"], smooth_score))
        return updated_tracks
