import time
import cv2

from .helper import safe_division

cache = {}


def dummy_function(frame, frame_metadata: dict):
    time.sleep(1)


def show_fps(frame, frame_metadata: dict):
    timings = frame_metadata["timings"]["frame"]
    fps = safe_division(1, (timings[1] - timings[0]))

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def show_smooth_fps(frame, frame_metadata: dict, smoothing_factor=0.95):
    timings = frame_metadata["timings"]["frame"]
    dt = max(timings[1] - timings[0], 1e-6)

    cache["frames_shown"] = cache.get("frames_shown", 0) + 1

    if cache["frames_shown"] < 30:
        show_fps(frame, frame_metadata)
        return

    if "last_dt" not in cache:
        cache["last_dt"] = dt
    else:
        cache["last_dt"] = (
            smoothing_factor * cache["last_dt"] + (1 - smoothing_factor) * dt
        )

    smooth_fps = 1.0 / cache["last_dt"]

    cv2.putText(
        frame,
        f"FPS: {smooth_fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


_face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def draw_head_box(frame, frame_metadata: dict):
    """Detect heads and draw boxes directly on bgr_frame (in place)."""
    bgr_frame = frame
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    faces = _face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    for x, y, w, h in faces:
        cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
