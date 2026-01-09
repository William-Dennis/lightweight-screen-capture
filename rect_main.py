from lightweight_screen_capture.display import display
from lightweight_screen_capture.functions import show_smooth_fps
from lightweight_screen_capture.screen_capture import capture_window
from lightweight_screen_capture.rectangle_detector import detect_rectangles

from functools import partial

# detect_rectangles = partial(detect_rectangles, min_percentage=0.05, max_percentage=0.8, show_confidence=False)

if __name__ == "__main__":
    display(
        [show_smooth_fps, detect_rectangles],
        source=capture_window,
        source_kwargs={"window_name": "Task Manager"},
    )