"""Visual region detection using YOLOv8 for image/video/ad detection."""

import numpy as np
from typing import List, Tuple

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    YOLO = None
    torch = None


class RegionDetector:
    """Detects visual regions using YOLOv8 nano model with GPU acceleration."""

    def __init__(
        self, model_size: str = "n", conf_threshold: float = 0.25, device: str = None
    ):
        """Initialize YOLOv8 detector with GPU support."""
        if YOLO is None:
            raise ImportError("ultralytics package required")
        if torch is None:
            raise ImportError("torch package required")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.model.to(self.device)
        self.conf_threshold = conf_threshold
        print(f"✓ RegionDetector initialized on device: {self.device}")

    def detect_regions(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int, float, int]]:
        """Detect regions in frame, returns list of (x1, y1, x2, y2, conf, cls)."""
        results = self.model(frame, verbose=False, conf=self.conf_threshold)
        regions = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    regions.append((int(x1), int(y1), int(x2), int(y2), conf, cls))
        return regions
