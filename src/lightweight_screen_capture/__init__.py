"""Lightweight screen capture library with AI content detection."""

from .ai_detector import detect_ai_content
from .camera_capture import capture_camera
from .screen_capture import capture_screen, capture_window
from .display import display
from .functions import show_fps, show_smooth_fps, draw_head_box

__all__ = [
    "detect_ai_content",
    "capture_camera",
    "capture_screen",
    "capture_window",
    "display",
    "show_fps",
    "show_smooth_fps",
    "draw_head_box",
]
