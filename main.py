from lightweight_screen_capture.display import display
from lightweight_screen_capture.functions import show_smooth_fps
from lightweight_screen_capture.screen_capture import capture_window

if __name__ == "__main__":
    display(
        [show_smooth_fps],
        source=capture_window,
        source_kwargs={"window_name": "Task Manager"},
    )
