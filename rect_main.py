from lightweight_screen_capture.display import display
from lightweight_screen_capture.functions import show_smooth_fps
from lightweight_screen_capture.screen_capture import capture_window, capture_screen
from lightweight_screen_capture.rectangle_detector import detect_rectangles

from functools import partial

source_function = partial(capture_window, window_name="Task Manager")
source_function = partial(capture_screen, monitor=2)

detect_rectangles = partial(
    detect_rectangles,
    min_percentage=0.01,)


if __name__ == "__main__":
    display(
        [show_smooth_fps, detect_rectangles],
        source=source_function,
    )