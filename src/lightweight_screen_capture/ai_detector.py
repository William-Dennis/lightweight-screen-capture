"""AI content detection pipeline integrating all components."""

import cv2
import numpy as np
from typing import Optional

try:
    import torch
except ImportError:
    torch = None

from .region_detector import RegionDetector
from .region_tracker import RegionTracker
from .alert_handler import AlertHandler


class AIContentPipeline:
    """Main pipeline for detecting AI-generated content in screen captures."""

    def __init__(
        self,
        enable_detector: bool = True,
        model_size: str = "n",
        output_dir: str = "ai_content_alerts",
        device: str = None,
    ):
        """Initialize AI content detection pipeline with GPU acceleration."""
        self.enable_detector = enable_detector
        self.detector: Optional[RegionDetector] = None
        self.tracker = RegionTracker(
            iou_threshold=0.3, history_size=5, score_threshold=0.6
        )
        self.alert_handler = AlertHandler(output_dir=output_dir)
        self.alerted_tracks = set()
        self._check_gpu()
        if enable_detector:
            try:
                self.detector = RegionDetector(
                    model_size=model_size, conf_threshold=0.25, device=device
                )
            except ImportError as e:
                print(f"⚠️  Detection disabled: {e}")
                self.enable_detector = False
            except Exception as e:
                print(f"⚠️  Detector initialization failed: {e}")
                self.enable_detector = False

    def _check_gpu(self):
        """Check and report GPU availability."""
        if torch is None:
            print("⚠️  PyTorch not available, CPU mode only")
            return
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA GPU detected: {gpu_name}")
            print(f"✓ CUDA version: {torch.version.cuda}")
        else:
            print("⚠️  No CUDA GPU detected, using CPU")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame through the pipeline, returns annotated frame."""
        if not self.enable_detector or self.detector is None:
            return frame
        try:
            detections = self.detector.detect_regions(frame)
            tracked = self.tracker.update(detections)
            for track_id, box, score in tracked:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                label = f"AI:{score:.2f}"
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                if track_id not in self.alerted_tracks:
                    saved_path = self.alert_handler.save_alert(
                        frame, box, score, track_id
                    )
                    self.alert_handler.print_alert(track_id, score, saved_path)
                    self.alerted_tracks.add(track_id)
        except Exception as e:
            print(f"⚠️  Frame processing error: {e}")
        return frame


_pipeline: Optional[AIContentPipeline] = None


def detect_ai_content(frame: np.ndarray, frame_metadata: dict):
    """Function to detect AI content in frame (compatible with display function)."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AIContentPipeline(enable_detector=True, model_size="n", device=None)
    processed = _pipeline.process_frame(frame.copy())
    frame[:] = processed
