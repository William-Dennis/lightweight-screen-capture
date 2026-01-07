import time
import cv2

from .helper import safe_division


def dummy_function(frame_data: tuple):
    time.sleep(1)


def show_fps(frame_data: tuple):
    timings = frame_data[0]
    fps = safe_division(1, (timings[1] - timings[0]))

    cv2.putText(
        frame_data[1],
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
