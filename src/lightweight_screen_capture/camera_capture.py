import cv2
import numpy as np

# Initialize camera once
_cap = cv2.VideoCapture(0)
if not _cap.isOpened():
    raise RuntimeError("Cannot open camera")


def capture_camera(flip: bool = True) -> np.ndarray:
    """Capture one frame from the default camera and return as BGR numpy array.

    Args:
        flip: If True, horizontally flip the frame for a mirrored view.

    Returns:
        BGR frame as a numpy array.
    """
    ret, frame = _cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame from camera")
    if flip:
        frame = cv2.flip(frame, 1)  # 1 means horizontal flip
    return frame
