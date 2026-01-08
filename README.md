# Lightweight Screen Capture

High-performance screen capture library with AI-powered content detection.

## Features

- **Screen Capture**: Capture screen at 5-10 FPS with minimal overhead
- **Camera Capture**: Capture from webcam with optional mirroring
- **AI Content Detection**: Real-time AI-generated content detection pipeline
- **GPU Acceleration**: CUDA support for YOLOv8 inference
- **Modular Design**: Composable functions for custom processing pipelines

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

### AI Content Detection

```python
from lightweight_screen_capture import display, capture_screen, detect_ai_content

display(functions=[detect_ai_content], source=capture_screen)
```

## AI Content Detection Pipeline

The AI detection pipeline implements a multi-stage approach:

1. **Visual Region Detection** - YOLOv8 detects image/video/ad regions
2. **Region Extraction** - Crops and resizes detected regions
3. **Temporal Tracking** - Tracks regions over time with IOU matching
4. **Score Smoothing** - Reduces false positives with history-based smoothing
5. **Alert System** - Prints alerts and saves flagged images with metadata

### Configuration

```python
from lightweight_screen_capture.ai_detector import detect_ai_content

# Customize detection parameters
display(
    functions=[lambda f, m: detect_ai_content(f, m, model_size='s', output_dir='alerts')],
    source=capture_screen
)
```

Parameters:
- `model_size`: YOLOv8 model size (`'n'`, `'s'`, `'m'`) - default `'n'`
- `output_dir`: Directory for saved alerts - default `'ai_content_alerts'`

### Output

When AI content is detected:
- Console alert with track ID and confidence score
- Image saved to `{output_dir}/alert_TIMESTAMP_TRACKID.jpg`
- Metadata saved to `{output_dir}/alert_TIMESTAMP_TRACKID.json`

See [docs/AI_DETECTION.md](docs/AI_DETECTION.md) for detailed documentation.

## Development

Format and lint code:

```bash
uv ruff format
uv ruff check --fix
```

## License

See [LICENSE](LICENSE) file for details.