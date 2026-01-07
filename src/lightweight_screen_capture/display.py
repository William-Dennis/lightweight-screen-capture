import cv2
from .capture import capture_screen

DEFAULT_FPS = 40.0


def show_screen(function=None, fps: float = DEFAULT_FPS, show_fps=False):
    """Continuously capture and display the screen until 'q' is pressed."""
    monitor = 1
    window_name = "Screen Capture"

    # Create the window once before the loop
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        bgra_frame = capture_screen(monitor)
        bgr_frame = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2BGR)
        cv2.imshow(window_name, bgr_frame)

        if function:
            function(bgra_frame)

        # Check for 'q' key press
        if cv2.waitKey(int(1000 // fps)) & 0xFF == ord("q"):
            break

        # Check if window was closed by user
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
