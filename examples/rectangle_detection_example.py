"""Example showing rectangle detection on screen capture."""

from lightweight_screen_capture import (
    display,
    capture_screen,
    show_smooth_fps,
    detect_rectangles,
)
import cv2


def draw_rectangles(frame, frame_metadata):
    """Detect and draw rectangles on frame."""
    rectangles = detect_rectangles(frame, min_area=10000, max_area=500000)

    for x1, y1, x2, y2, confidence in rectangles:
        # Draw rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw confidence label
        label = f"{confidence * 100:.1f}%"
        cv2.putText(
            frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )


if __name__ == "__main__":
    print("Rectangle Detection Demo")
    print("Press 'q' to quit")

    display(
        functions=[show_smooth_fps, draw_rectangles],
        source=capture_screen,
    )
