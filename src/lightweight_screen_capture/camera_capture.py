import cv2
import numpy as np

# Initialize camera once
_cap = cv2.VideoCapture(0)
if not _cap.isOpened():
    raise RuntimeError("Cannot open camera")

def capture_camera() -> np.ndarray:
    """Capture one frame from the default camera and return as BGR numpy array."""
    ret, frame = _cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame from camera")
    return frame  # Already BGR, dtype=uint8, shape=(H, W, 3)
