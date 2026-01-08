import cv2
import numpy as np

# Initialize camera once (lazy initialization)
_cap = None


def _get_camera():
    """Get or initialize camera capture."""
    global _cap
    if _cap is None:
        _cap = cv2.VideoCapture(0)
        if not _cap.isOpened():
            raise RuntimeError("Cannot open camera")
    return _cap


def capture_camera(flip: bool = True) -> np.ndarray:
    """Capture one frame from the default camera and return as BGR numpy array."""
    cap = _get_camera()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame from camera")
    if flip:
        frame = cv2.flip(frame, 1)
    return frame
