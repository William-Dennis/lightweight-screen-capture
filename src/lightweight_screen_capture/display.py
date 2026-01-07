import cv2
from .capture import capture_screen
import time


def show_screen(functions=[]):
    """Continuously capture and display the screen until 'q' is pressed."""
    monitor = 1
    window_name = "Screen Capture"

    # Create the window once before the loop
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_timings = {0: 0, 1: 0}
    while True:
        bgra_frame = capture_screen(monitor)
        bgr_frame = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2BGR)

        if len(functions) > 0:
            for f in functions:
                f((frame_timings, bgr_frame))

        cv2.imshow(window_name, bgr_frame)

        # Check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Check if window was closed by user
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # add timings
        frame_timings[0] = frame_timings[1]
        frame_timings[1] = time.perf_counter()

    cv2.destroyAllWindows()
