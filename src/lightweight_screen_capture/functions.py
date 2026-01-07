import time
import cv2

from .helper import safe_division

cache = {"last_fps": 30}


def dummy_function(frame_data: tuple):
    time.sleep(1)


def show_fps(frame_data: tuple):
    timings = frame_data[0]
    fps = safe_division(1, (timings[1] - timings[0]))

    cv2.putText(
        frame_data[1],
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def show_smooth_fps(frame_data: tuple, smoothing_factor=0.99):
    timings = frame_data[0]
    fps = safe_division(1, (timings[1] - timings[0]))
    smooth_fps = smoothing_factor * cache["last_fps"] + (1 - smoothing_factor) * fps
    cache["last_fps"] = smooth_fps

    cv2.putText(
        frame_data[1],
        f"FPS: {int(smooth_fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
