"""Example demonstrating square box detector for ads and galleries."""

from lightweight_screen_capture import display, capture_screen, show_smooth_fps
from functools import partial

# Use square detector for low-latency detection of ads/galleries
def detect_squares(frame, metadata):
    """Wrapper to enable square box detector."""
    from lightweight_screen_capture.ai_detector import detect_ai_content
    detect_ai_content(frame, metadata, use_square_detector=True)


if __name__ == "__main__":
    print("Starting square box detector...")
    print("This will detect square/rectangular regions like ads and image galleries")
    print("Press 'q' to quit")
    
    display(
        functions=[show_smooth_fps, detect_squares],
        source=capture_screen,
    )
