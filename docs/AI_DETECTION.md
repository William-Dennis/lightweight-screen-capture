# AI Content Detection Pipeline

This module implements a real-time AI-generated content detection pipeline for screen captures.

## Features

- **Screen Capture**: Captures frames at 5-10 FPS
- **Visual Region Detection**: Uses YOLOv8 (nano/small) to detect image/video/ad regions
- **Region Extraction**: Crops and resizes detected regions
- **Batch AI Detection**: Processes regions for AI content
- **Temporal Smoothing**: Tracks regions over time to reduce false positives
- **Alert System**: Prints alerts and saves flagged images with metadata

## Architecture

```
Screen Capture (5-10 FPS)
    ↓
Visual Region Detection (YOLOv8)
    ↓
Region Extraction & Cropping
    ↓
Batch AI Image Detection
    ↓
Temporal Smoothing & Tracking
    ↓
Alert / Overlay (Print + Save)
```

## Usage

```python
from lightweight_screen_capture import display, capture_screen, detect_ai_content, show_smooth_fps

# Run with AI detection enabled
display(
    functions=[show_smooth_fps, detect_ai_content],
    source=capture_screen,
)
```

## Configuration

The pipeline is modular and can be configured:

- **Model Size**: Choose between YOLOv8 nano (`n`), small (`s`), or medium (`m`)
- **IOU Threshold**: Adjust region tracking sensitivity (default: 0.3)
- **Score Threshold**: Set confidence threshold for alerts (default: 0.6)
- **History Size**: Number of frames to smooth over (default: 5)
- **Output Directory**: Where to save flagged images (default: `ai_content_alerts/`)

## Performance

- Optimized for 5-10 FPS requirement
- Uses YOLOv8 nano model by default for lightweight inference
- Modular design allows component replacement
- Each module < 300 lines, functions < 30 lines

## Output

When AI content is detected:
1. Console alert is printed with track ID and confidence score
2. Image is saved to `ai_content_alerts/alert_TIMESTAMP_TRACKID.jpg`
3. Metadata saved to `ai_content_alerts/alert_TIMESTAMP_TRACKID.json`

## Dependencies

- ultralytics >= 8.0.0 (YOLOv8)
- torch >= 2.6.0
- pillow >= 10.3.0
- opencv-python >= 4.11.0.86
- numpy >= 2.4.0
