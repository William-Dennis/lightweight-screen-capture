import numpy as np

try:
    import mss
except ImportError:
    mss = None

try:
    import win32gui
except ImportError:
    win32gui = None

_sct = None


def _get_mss():
    """Get or initialize mss screen capture."""
    global _sct
    if _sct is None:
        if mss is None:
            raise ImportError("mss package required for screen capture")
        _sct = mss.mss()
    return _sct


def capture_screen(monitor=1) -> np.ndarray:
    """Capture screen, returns BGRA image as numpy array."""
    sct = _get_mss()
    img = sct.grab(sct.monitors[monitor])
    return np.asarray(img)


def capture_window(window_name: str) -> np.ndarray:
    """Capture a specific window by its title, returns BGRA image as numpy array."""
    if win32gui is None:
        raise RuntimeError("win32gui not available, install pywin32 package")
    sct = _get_mss()
    hwnd = win32gui.FindWindow(None, window_name)
    if hwnd == 0:
        raise RuntimeError(f"Window '{window_name}' not found")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    region = {
        "left": left,
        "top": top,
        "width": right - left,
        "height": bottom - top,
    }
    img = sct.grab(region)
    return np.asarray(img)
