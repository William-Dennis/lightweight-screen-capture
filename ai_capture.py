from lightweight_screen_capture import display, capture_screen, detect_ai_content, capture_window
from lightweight_screen_capture import show_smooth_fps
from functools import partial

source_func = partial(capture_window, window_name = "Task Manager")

# Basic usage with defaults (YOLOv8n, CPU/GPU auto)
display(functions=[show_smooth_fps, detect_ai_content], source=source_func)

# # Customized
# display(
#     functions=[lambda f, m: detect_ai_content(f, m, model_size='s', output_dir='alerts')],
#     source=source_func
# )