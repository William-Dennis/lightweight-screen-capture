from lightweight_screen_capture.camera_capture import capture_camera
from lightweight_screen_capture.display import display
from lightweight_screen_capture.functions import show_smooth_fps, draw_head_box

if __name__ == "__main__":
    display(
        [show_smooth_fps, draw_head_box],
        source=capture_camera,
        # source_kwargs={"window_name": "Task Manager"},
    )
