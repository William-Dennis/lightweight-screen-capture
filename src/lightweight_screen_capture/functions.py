import time
import numpy as np

from .helper import safe_division


def dummy_function(frame_data: np.ndarray):
    time.sleep(1)


def show_fps(frame_data: np.ndarray):
    timings = frame_data[0]
    print("fps", safe_division(1, (timings[1] - timings[0])))
