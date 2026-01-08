# AI Content Detection - Configuration Guide

## Overview

The AI content detection pipeline is now fully customizable with multiple parameters to control detection, display, and saving behavior.

## Key Features

- **Display all detected boxes** with AI confidence percentage
- **Color-coded boxes**: 
  - 🟢 Green = High confidence (>90%, will be saved)
  - 🔴 Red = Lower confidence (displayed but not saved)
- **Save only high-confidence detections** (default: >90%)
- **Fully configurable** thresholds and behavior

## Usage Examples

### Basic Usage (Default Settings)

```python
from lightweight_screen_capture import display, capture_screen, detect_ai_content

# Uses square detector, shows all boxes, saves only >90%
display(functions=[detect_ai_content], source=capture_screen)
```

### Custom Save Threshold

```python
# Save only detections with >95% confidence
display(
    functions=[lambda f, m: detect_ai_content(f, m, save_threshold=0.95)],
    source=capture_screen
)
```

### Filter Low-Confidence Detections

```python
# Only display and track detections with >50% confidence
display(
    functions=[lambda f, m: detect_ai_content(f, m, min_score=0.5)],
    source=capture_screen
)
```

### Use YOLOv8 Instead of Square Detector

```python
# Use YOLOv8 small model for detection
display(
    functions=[lambda f, m: detect_ai_content(
        f, m, 
        use_square_detector=False,
        model_size='s'  # 's', 'm', 'l', 'x' for more powerful models
    )],
    source=capture_screen
)
```

### Full Customization

```python
# Complete control over all parameters
display(
    functions=[lambda f, m: detect_ai_content(
        f, m,
        model_size='m',              # Medium YOLO model
        output_dir='my_alerts',      # Custom save directory
        use_square_detector=False,   # Use YOLO instead of square detector
        save_threshold=0.85,         # Save at 85% confidence
        min_score=0.3,               # Show boxes above 30%
    )],
    source=capture_screen
)
```

## Configuration Parameters

### `save_threshold` (default: 0.9)
- **Type**: float (0.0 - 1.0)
- **Description**: Minimum confidence required to save detection
- **Example**: `save_threshold=0.95` saves only 95%+ confidence detections

### `min_score` (default: 0.0)
- **Type**: float (0.0 - 1.0)
- **Description**: Minimum confidence to display/track detection
- **Example**: `min_score=0.5` filters out low-confidence noise

### `display_all` (default: True)
- **Type**: bool
- **Description**: Whether to display all tracked detections
- **Example**: `display_all=False` shows only high-confidence detections

### `use_square_detector` (default: True)
- **Type**: bool
- **Description**: Use fast square detector (True) or YOLOv8 (False)
- **Example**: `use_square_detector=False` enables YOLO

### `model_size` (default: "n")
- **Type**: str ("n", "s", "m", "l", "x")
- **Description**: YOLOv8 model size (larger = more accurate but slower)
- **Options**:
  - `"n"` - Nano (fastest, 2-5 FPS CPU)
  - `"s"` - Small (balanced)
  - `"m"` - Medium (more accurate)
  - `"l"` - Large (high accuracy)
  - `"x"` - Extra large (best accuracy, slowest)

### `output_dir` (default: "ai_content_alerts")
- **Type**: str
- **Description**: Directory to save alert images and metadata
- **Example**: `output_dir='custom_alerts'`

## Visual Feedback

### Box Colors
- **Green box + background**: Detection will be saved (≥ save_threshold)
- **Red box + background**: Detection displayed but not saved (< save_threshold)

### Labels
- Format: `AI: XX.X%` where XX.X is the confidence percentage
- White text on colored background for better visibility

## Performance Tips

1. **For speed**: Use square detector with `min_score=0.5` to filter noise
2. **For accuracy**: Use YOLOv8 with larger model size (`model_size='m'` or `'l'`)
3. **For storage**: Increase `save_threshold` to save only very confident detections
4. **For monitoring**: Keep `save_threshold=0.9` and `min_score=0.0` (default)

## Examples by Use Case

### Gaming/Streaming
```python
# Show all, save only very confident AI content
detect_ai_content(f, m, save_threshold=0.95, min_score=0.0)
```

### Security Monitoring
```python
# More sensitive, save more detections
detect_ai_content(f, m, save_threshold=0.75, min_score=0.5)
```

### Content Moderation
```python
# Use powerful model, save high-confidence only
detect_ai_content(f, m, use_square_detector=False, model_size='l', save_threshold=0.9)
```

### Research/Analysis
```python
# Save everything above threshold for later analysis
detect_ai_content(f, m, save_threshold=0.7, min_score=0.3)
```

## Output Files

When a detection is saved (≥ save_threshold):
- **Image**: `{output_dir}/alert_YYYYMMDD_HHMMSS_FFFFFF_TRACKID.jpg`
- **Metadata**: `{output_dir}/alert_YYYYMMDD_HHMMSS_FFFFFF_TRACKID.json`

Metadata includes:
- Timestamp
- Track ID
- Confidence score
- Bounding box coordinates
- Image file path
