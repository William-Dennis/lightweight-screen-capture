# Rectangle Detection

Simple OpenCV-based rectangle detection for finding rectangular regions in images.

## Usage

```python
from lightweight_screen_capture import detect_rectangles

rectangles = detect_rectangles(
    frame,           # Input BGR image
    min_area=10000,  # Minimum rectangle area (pixels)
    max_area=500000  # Maximum rectangle area (pixels)
)

# Returns: [(x1, y1, x2, y2, confidence), ...]
for x1, y1, x2, y2, confidence in rectangles:
    print(f"Found rectangle with {confidence*100:.1f}% confidence")
```

## Algorithm

1. Convert to grayscale
2. Apply Gaussian blur
3. Canny edge detection
4. Morphological closing to connect edges
5. Find contours
6. Filter by:
   - Area (min_area to max_area)
   - Shape (must have 4 corners)
   - Aspect ratio (0.1 to 10.0)
7. Calculate confidence based on fill ratio
8. Remove overlapping detections (IOU > 0.5)

## Confidence Score

Confidence (0.0-1.0) represents how well the contour fills its bounding box:
- 1.0 = Perfect rectangle
- 0.7+ = Good rectangle
- <0.5 = Irregular shape

## Performance

- ~10-30 FPS on CPU (depends on image size and complexity)
- No GPU or ML models required
- Low latency (~10-50ms per frame)
