import cv2
import time


def display(source, functions=[], source_kwargs={}):
    """Continuously capture and display the screen until 'q' is pressed."""
    window_name = "Screen Capture"

    # DISPLAY_W, DISPLAY_H = 1000, 600

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.resizeWindow(window_name, DISPLAY_W, DISPLAY_H)

    frame_metadata = {"timings": {"frame": [0, 0]}}
    while True:
        bgra_frame = source(**source_kwargs)
        bgr_frame = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2BGR)

        if len(functions) > 0:
            for f in functions:
                f(bgr_frame, frame_metadata)

        cv2.imshow(window_name, bgr_frame)

        # Check for 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        # Check if window was closed by user
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # add timings
        frame_metadata["timings"]["frame"][0] = frame_metadata["timings"]["frame"][1]
        frame_metadata["timings"]["frame"][1] = time.perf_counter()

    cv2.destroyAllWindows()
