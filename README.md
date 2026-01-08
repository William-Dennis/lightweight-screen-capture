# Lightweight Screen Capture

High-performance screen capture library with rectangle detection.

## Features

- **Screen Capture**: Capture screen at 5-10 FPS with minimal overhead
- **Camera Capture**: Capture from webcam with optional mirroring
- **Rectangle Detection**: OpenCV-based detection of rectangular regions with confidence scores

## Installation

Requires Python 3.11+. Install with uv:

```bash
uv sync
```

## Quick Start

### Basic Screen Capture

```python
from lightweight_screen_capture import display, capture_screen, show_smooth_fps

display(functions=[show_smooth_fps], source=capture_screen)
```

### Rectangle Detection

```python
from lightweight_screen_capture import detect_rectangles
import cv2

frame = cv2.imread('image.jpg')
rectangles = detect_rectangles(frame, min_area=10000, max_area=500000)

for x1, y1, x2, y2, confidence in rectangles:
    print(f"Rectangle at ({x1},{y1})-({x2},{y2}) confidence: {confidence:.2f}")
```

## Rectangle Detection

The `detect_rectangles` function uses OpenCV edge detection and contour analysis to find rectangular regions:

```python
detect_rectangles(
    frame,           # Input image (BGR)
    min_area=10000,  # Minimum area in pixels
    max_area=500000  # Maximum area in pixels
) -> List[Tuple[int, int, int, int, float]]  # [(x1, y1, x2, y2, confidence), ...]
```

**Returns:** List of (x1, y1, x2, y2, confidence) tuples where confidence is 0.0-1.0

## Development

Format and lint code:

```bash
uv ruff format
uv ruff check --fix
```

## License

See [LICENSE](LICENSE) file for details.
