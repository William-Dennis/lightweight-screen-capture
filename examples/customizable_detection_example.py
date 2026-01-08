"""Example showing customizable AI detection with visual feedback."""

from lightweight_screen_capture import display, capture_screen, show_smooth_fps
from functools import partial


def detect_with_custom_settings(frame, metadata):
    """AI detection with custom save threshold and display settings."""
    from lightweight_screen_capture.ai_detector import detect_ai_content
    
    # Display all detections, save only >90% confidence
    # Green boxes = will save, Red boxes = display only
    detect_ai_content(
        frame,
        metadata,
        save_threshold=0.9,      # Save only ≥90% confidence
        min_score=0.0,           # Show all detections
        use_square_detector=True, # Use fast square detector
        output_dir="ai_content_alerts"
    )


if __name__ == "__main__":
    print("="*60)
    print("AI Content Detection - Visual Feedback Demo")
    print("="*60)
    print("\nBox Colors:")
    print("  🟢 GREEN = High confidence (≥90%, will be saved)")
    print("  🔴 RED   = Lower confidence (displayed only)")
    print("\nPress 'q' to quit\n")
    print("="*60)
    
    display(
        functions=[show_smooth_fps, detect_with_custom_settings],
        source=capture_screen,
    )
