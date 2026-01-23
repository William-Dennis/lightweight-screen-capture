import mss
import numpy as np
import sys

if sys.platform == "win32":
    import win32gui  # type: ignore

    WINDOWS = True


class ScreenCapturer:
    def __init__(self):
        # The resource is only initialized when the class is instantiated
        self.sct = mss.mss()

    def capture_screen(self, monitor: int = 1) -> np.ndarray:
        if monitor >= len(self.sct.monitors):
            raise IndexError(f"Monitor {monitor} not found.")

        img = self.sct.grab(self.sct.monitors[monitor])
        return np.asarray(img)

    def capture_window(self, window_name: str) -> np.ndarray:
        if not WINDOWS:
            raise NotImplementedError("This feature is only available for windows")
        hwnd = win32gui.FindWindow(None, window_name)
        if hwnd == 0:
            raise RuntimeError(f"Window '{window_name}' not found")

        left, top, right, bottom = win32gui.GetWindowRect(hwnd)

        region = {
            "left": left,
            "top": top,
            "width": max(0, right - left),
            "height": max(0, bottom - top),
        }

        img = self.sct.grab(region)
        return np.asarray(img)

    def close(self):
        self.sct.close()
