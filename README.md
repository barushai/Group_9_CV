# Traffic Analysis System

A traffic analysis system for dashcam videos, integrating core Computer Vision components including image preprocessing, object detection, multi-object tracking, and real-time visualization.

## Main Components

- `main_enhanced.py`: full implementation of the processing pipeline.
- `main.py`: baseline version for processing and visualization.
- `get_data.py`: utility script for preparing video data.
- `report/`: technical report and supporting materials.

## Tech Stack

- Python 3.12
- OpenCV
- PyTorch
- Ultralytics YOLOv8
- CustomTkinter

## Quick Start

```bash
pip install -r requirements.txt
python main_enhanced.py
```

## Notes

This project is configured primarily for CPU execution. Large files (e.g., raw videos, model weights) should be managed separately and should not be pushed directly to a public repository.
