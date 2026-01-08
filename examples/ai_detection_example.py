"""Example demonstrating AI content detection pipeline."""

from lightweight_screen_capture import display, capture_screen, detect_ai_content, show_smooth_fps


if __name__ == "__main__":
    display(
        functions=[show_smooth_fps, detect_ai_content],
        source=capture_screen,
    )
