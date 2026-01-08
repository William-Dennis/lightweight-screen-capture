"""AI content detection pipeline integrating all components."""

import cv2
import numpy as np
from typing import Optional, List, Tuple

try:
    import torch
except ImportError:
    torch = None

from .region_detector import RegionDetector
from .region_tracker import RegionTracker
from .alert_handler import AlertHandler
from .square_box_detector import SquareBoxDetector


class AIContentPipeline:
    """Main pipeline for detecting AI-generated content in screen captures."""

    def __init__(
        self,
        enable_detector: bool = True,
        model_size: str = "n",
        output_dir: str = "ai_content_alerts",
        device: str = None,
        use_square_detector: bool = True,
        save_threshold: float = 0.9,  # Only save detections above 90%
        display_all: bool = True,  # Display all detections regardless of score
        min_score: float = 0.0,  # Minimum score to display
    ):
        """Initialize AI content detection pipeline with GPU acceleration."""
        self.enable_detector = enable_detector
        self.detector: Optional[RegionDetector] = None
        self.square_detector: Optional[SquareBoxDetector] = None
        self.use_square_detector = use_square_detector
        self.save_threshold = save_threshold
        self.display_all = display_all
        self.min_score = min_score
        self.tracker = RegionTracker(
            iou_threshold=0.3, history_size=5, score_threshold=min_score
        )
        self.alert_handler = AlertHandler(output_dir=output_dir)
        self.alerted_tracks = set()
        self._check_gpu()
        if enable_detector:
            self._init_detector(use_square_detector, model_size, device)

    def _init_detector(self, use_square: bool, model_size: str, device: str):
        """Initialize either square or YOLO detector."""
        if use_square:
            self.square_detector = SquareBoxDetector()
            print("✓ Using square box detector (low-latency cv2)")
        else:
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

    def _get_detections(self, frame: np.ndarray) -> List:
        """Get detections from appropriate detector."""
        if self.square_detector is not None:
            return self.square_detector.detect_square_regions(frame)
        elif self.detector is not None:
            return self.detector.detect_regions(frame)
        return []

    def _draw_box(
        self, frame: np.ndarray, box: Tuple, score: float, should_save: bool = False
    ):
        """Draw detection box and label on frame with color coding."""
        x1, y1, x2, y2 = box
        # Color: Green if will be saved (>90%), Red otherwise
        color = (0, 255, 0) if should_save else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        # Display AI percentage
        label = f"AI: {score * 100:.1f}%"
        # Add background to text for better visibility
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(
            frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1
        )
        cv2.putText(
            frame,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process single frame through the pipeline, returns annotated frame."""
        if not self.enable_detector:
            return frame
        try:
            detections = self._get_detections(frame)
            tracked = self.tracker.update(detections)
            active_track_ids = set(self.tracker.tracked_regions.keys())
            self.alerted_tracks &= active_track_ids

            for track_id, box, score in tracked:
                # Determine if this should be saved (score > save_threshold)
                should_save = score >= self.save_threshold

                # Draw all boxes with their AI percentage
                self._draw_box(frame, box, score, should_save)

                # Only save and alert for high-confidence detections
                if should_save and track_id not in self.alerted_tracks:
                    saved_path = self.alert_handler.save_alert(
                        frame, box, score, track_id
                    )
                    self.alert_handler.print_alert(track_id, score, saved_path)
                    self.alerted_tracks.add(track_id)
        except Exception as e:
            print(f"⚠️  Frame processing error: {e}")
        return frame


_pipeline: Optional[AIContentPipeline] = None


def _get_or_create_pipeline(
    model_size: str,
    output_dir: str,
    use_square_detector: bool,
    save_threshold: float,
    display_all: bool,
    min_score: float,
) -> AIContentPipeline:
    """Get or create global pipeline instance."""
    global _pipeline
    if _pipeline is None:
        _pipeline = AIContentPipeline(
            enable_detector=True,
            model_size=model_size,
            device=None,
            output_dir=output_dir,
            use_square_detector=use_square_detector,
            save_threshold=save_threshold,
            display_all=display_all,
            min_score=min_score,
        )
    return _pipeline


def detect_ai_content(
    frame: np.ndarray,
    frame_metadata: dict,
    model_size: str = "n",
    output_dir: str = "ai_content_alerts",
    use_square_detector: bool = True,
    save_threshold: float = 0.9,
    display_all: bool = True,
    min_score: float = 0.0,
):
    """Detect AI content, display all boxes with AI%, save only high-confidence (>90%)."""
    pipeline = _get_or_create_pipeline(
        model_size,
        output_dir,
        use_square_detector,
        save_threshold,
        display_all,
        min_score,
    )
    processed = pipeline.process_frame(frame.copy())
    frame[:] = processed
