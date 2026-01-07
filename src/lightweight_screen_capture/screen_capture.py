import mss
import numpy as np
import win32gui

_sct = mss.mss()


def capture_screen(monitor=1) -> np.ndarray:
    img = _sct.grab(_sct.monitors[monitor])
    return np.asarray(img)


def capture_window(window_name: str) -> np.ndarray:
    """
    Capture a specific window by its title.
    Returns BGRA image as numpy array.
    """

    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        raise RuntimeError(f"Window '{window_name}' not found")

    # Get window rect (includes borders)
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)

    region = {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
    }

    img = _sct.grab(region)
    return np.asarray(img)
