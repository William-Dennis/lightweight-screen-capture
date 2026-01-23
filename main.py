from lightweight_screen_capture.camera_capture import CameraManager
from lightweight_screen_capture.display import display
from lightweight_screen_capture.functions import show_smooth_fps, draw_head_box

if __name__ == "__main__":
    camera = CameraManager()
    display(
        source=camera.capture_frame,
        functions=[show_smooth_fps, draw_head_box],
        # source_kwargs={"window_name": "Task Manager"},
    )
