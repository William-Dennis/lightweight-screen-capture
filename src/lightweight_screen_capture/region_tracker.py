"""Temporal tracking and smoothing for detected regions."""

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque


class RegionTracker:
    """Tracks regions over time and smooths detection scores."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        history_size: int = 5,
        score_threshold: float = 0.0,  # Changed to 0.0 to return all tracks
        max_age: int = 5,
    ):
        """Initialize region tracker with IOU and history parameters."""
        self.iou_threshold = iou_threshold
        self.history_size = history_size
        self.score_threshold = score_threshold
        self.max_age = max_age
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

    def _find_best_match(self, box: Tuple, unmatched_tracks: set) -> Optional[int]:
        """Find best matching track for box from unmatched tracks only."""
        best_iou, best_id = 0, None
        for track_id in unmatched_tracks:
            track = self.tracked_regions[track_id]
            iou = self._compute_iou(box, track["box"])
            if iou > best_iou and iou > self.iou_threshold:
                best_iou, best_id = iou, track_id
        return best_id

    def _update_track(self, track_id: int, box: Tuple, conf: float, cls: int):
        """Update existing track with new detection."""
        track = self.tracked_regions[track_id]
        track["scores"].append(conf)
        track["box"] = box
        track["cls"] = cls
        track["age"] = 0

    def _create_track(self, box: Tuple, conf: float, cls: int) -> int:
        """Create new track."""
        track_id = self.next_id
        self.next_id += 1
        self.tracked_regions[track_id] = {
            "box": box,
            "scores": deque([conf], maxlen=self.history_size),
            "cls": cls,
            "age": 0,
        }
        return track_id

    def update(self, detections: List[Tuple]) -> List[Tuple[int, Tuple, float]]:
        """Update tracked regions and return (track_id, box, smooth_score)."""
        unmatched_tracks = set(self.tracked_regions.keys())
        matched_ids = set()
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            box = (x1, y1, x2, y2)
            track_id = self._find_best_match(box, unmatched_tracks)
            if track_id is not None:
                self._update_track(track_id, box, conf, cls)
                unmatched_tracks.remove(track_id)
            else:
                track_id = self._create_track(box, conf, cls)
            matched_ids.add(track_id)
        for track_id in unmatched_tracks:
            self.tracked_regions[track_id]["age"] += 1
        self.tracked_regions = {
            k: v for k, v in self.tracked_regions.items() if v["age"] < self.max_age
        }
        result = []
        for tid, t in self.tracked_regions.items():
            score = np.mean(t["scores"])
            if score >= self.score_threshold:
                result.append((tid, t["box"], score))
        return result
