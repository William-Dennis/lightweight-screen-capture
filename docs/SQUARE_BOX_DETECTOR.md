# Square Box Detector Usage

The square box detector is a lightweight, low-latency alternative to YOLOv8 for detecting square or rectangular regions like advertisements and image galleries.

## Features

- **Low Latency**: Uses only OpenCV (cv2) - no deep learning required
- **Fast**: Suitable for 10+ FPS on CPU
- **No GPU Required**: Pure CPU implementation
- **Detects**: Square/rectangular regions like ads, images in galleries, UI elements

## Usage

### With detect_ai_content function

```python
from lightweight_screen_capture import display, capture_screen, detect_ai_content

# Use square detector instead of YOLOv8
display(
    functions=[lambda f, m: detect_ai_content(f, m, use_square_detector=True)],
    source=capture_screen
)
```

### Direct usage

```python
from lightweight_screen_capture.square_box_detector import SquareBoxDetector
import cv2

# Initialize detector
detector = SquareBoxDetector(
    min_area=10000,      # Minimum box area (pixels)
    max_area=500000,     # Maximum box area (pixels)
    aspect_ratio_tolerance=0.3  # How square (0.3 means 0.7-1.3 aspect ratio)
)

# Detect regions
frame = cv2.imread('screenshot.png')
regions = detector.detect_square_regions(frame)

# Returns list of (x1, y1, x2, y2, confidence, class_id)
for x1, y1, x2, y2, conf, cls in regions:
    print(f"Square region at ({x1}, {y1}) - ({x2}, {y2}), conf: {conf:.2f}")
```

## Configuration

- **min_area** (default: 10000): Minimum pixel area for detection
- **max_area** (default: 500000): Maximum pixel area for detection
- **aspect_ratio_tolerance** (default: 0.3): How close to square (1.0 = perfect square)

## Performance Comparison

| Detector | FPS (CPU) | FPS (GPU) | Latency |
|----------|-----------|-----------|---------|
| YOLOv8n  | 2-5 FPS   | 10-20 FPS | ~100ms  |
| Square Box | 10-30 FPS | N/A       | ~10ms   |

## Use Cases

- Detecting advertisement boxes on websites
- Finding image thumbnails in galleries
- Locating square UI elements
- Quick region-of-interest detection
- Real-time applications where speed > accuracy

## Algorithm

1. Convert frame to grayscale
2. Apply Canny edge detection
3. Find contours
4. Filter by area and aspect ratio
5. Return bounding boxes

This is much faster than deep learning but less accurate for complex scenes.
