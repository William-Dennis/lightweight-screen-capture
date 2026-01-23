import cv2
import numpy as np


class CameraManager:
    def __init__(self, index: int = 0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {index}")

    def capture_frame(self, flip: bool = True) -> np.ndarray:
        """Captures a frame, optionally flips it, and returns it."""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

        if flip:
            frame = cv2.flip(frame, 1)
        return frame

    def release(self):
        """Properly closes the camera resource."""
        if self.cap.isOpened():
            self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
