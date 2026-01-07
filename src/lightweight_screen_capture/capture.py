import mss
import numpy as np


def capture_screen(monitor=1) -> np.ndarray:
    """Capture the specified monitor and return raw BGRA bytes as a NumPy array."""
    with mss.mss() as sct:
        img = sct.grab(sct.monitors[monitor])
        img_np = np.array(img)  # BGRA format
        return img_np
