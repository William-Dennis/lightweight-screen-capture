"""Lightweight screen capture library with rectangle detection."""

from .camera_capture import capture_camera
from .screen_capture import capture_screen, capture_window
from .display import display
from .functions import show_fps, show_smooth_fps, draw_head_box
from .rectangle_detector import detect_rectangles

__all__ = [
    "capture_camera",
    "capture_screen",
    "capture_window",
    "display",
    "show_fps",
    "show_smooth_fps",
    "draw_head_box",
    "detect_rectangles",
]
