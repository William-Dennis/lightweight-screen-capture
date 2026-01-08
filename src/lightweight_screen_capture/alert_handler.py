"""Alert handling and image saving for AI content detection."""

import cv2
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple


class AlertHandler:
    """Handles alerts and saves flagged images with metadata."""

    def __init__(self, output_dir: str = "ai_content_alerts"):
        """Initialize alert handler with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.alert_count = 0

    def save_alert(
        self, frame: np.ndarray, region_box: Tuple, score: float, track_id: int
    ) -> str:
        """Save flagged image with metadata, returns saved file path."""
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        filename = f"alert_{timestamp}_{track_id}"
        x1, y1, x2, y2 = region_box
        cropped = frame[y1:y2, x1:x2]
        img_path = self.output_dir / f"{filename}.jpg"
        cv2.imwrite(str(img_path), cropped)
        metadata = {
            "timestamp": now.isoformat(),
            "track_id": track_id,
            "confidence_score": float(score),
            "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            "image_path": str(img_path),
        }
        meta_path = self.output_dir / f"{filename}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        self.alert_count += 1
        return str(img_path)

    def print_alert(self, track_id: int, score: float, saved_path: str):
        """Print alert message to console."""
        print(
            f"⚠️  AI CONTENT DETECTED - Track ID: {track_id}, Score: {score:.2f}, Saved: {saved_path}"
        )
