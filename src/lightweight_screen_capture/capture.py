import mss
import numpy as np

_sct = mss.mss()


def capture_screen(monitor=1) -> np.ndarray:
    img = _sct.grab(_sct.monitors[monitor])
    return np.asarray(img)
