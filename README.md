# VidLens

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**VidLens** is an open-source video intelligence toolkit that runs deep learning computer vision models on any video file — no training required. Every model also supports fine-tuning on custom data when you're ready.

| Capability | Model | Pre-trained | Custom Training |
|---|---|---|---|
| Object Detection & Tracking | YOLOv8 | COCO 80 classes | Any classes |
| Face Detection & Anonymization | YOLOv8-face | ✓ | ✓ |
| Pose Estimation | YOLOv8-pose | 17 keypoints | ✓ |
| Scene Classification | CLIP | Zero-shot | ✓ |
| Highlight Reel Generation | CLIP | ✓ | — |

---

## Installation

### Linux — one command

```bash
git clone https://github.com/your-org/vidlens
cd vidlens
bash setup.sh
```

The setup script handles Python, FFmpeg, GPU detection, and model weight downloads automatically. New to the project? Start with the [Quickstart Guide](docs/quickstart.md).

### All platforms — pip

```bash
pip install vidlens[all]          # install everything
pip install vidlens[yolo]         # object detection, face, pose only
pip install vidlens[yolo,clip]    # + scene classification and highlights
```

---

## Quick Start

**CLI**

```bash
# Detect and track objects across frames
vidlens analyze myvideo.mp4 --lens objects

# Run multiple analyses in one pass
vidlens analyze myvideo.mp4 --lens objects --lens pose --lens scene

# Blur all faces (GDPR-friendly)
vidlens anonymize myvideo.mp4 --mode blur

# Generate a 60-second highlight reel
vidlens highlights match.mp4 --duration 60

# Generate highlights for specific events
vidlens highlights match.mp4 --labels "goal,tackle,celebration" --duration 90

# Train on your own data
vidlens train objects --data data/dataset.yaml --epochs 100

# Launch the web interface
vidlens ui
```

**Python API**

```python
from vidlens.pipeline import VideoPipeline

pipeline = VideoPipeline(
    video_path="myvideo.mp4",
    lenses=["objects", "pose"],
    output_dir="output/",
)
summary = pipeline.run()
# Saves: output/myvideo_annotated.mp4
# Saves: output/myvideo_report.json
```

```python
# Load a custom-trained model
from vidlens.lenses import ObjectDetectionLens

lens = ObjectDetectionLens(model_path="models/custom/best.pt")
```

```python
# Zero-shot scene detection — describe your categories in plain English
from vidlens.lenses import SceneClassificationLens

lens = SceneClassificationLens(labels=["fire", "smoke", "normal activity"])
```

---

## Custom Training

All lenses support fine-tuning on your own labeled data:

```bash
# Fine-tune object detector on custom classes
vidlens train objects --data configs/dataset_example.yaml --epochs 100 --device cuda

# Fine-tune face detector for a specific domain
vidlens train faces --data data/faces.yaml --epochs 50

# Fine-tune CLIP on custom image-text pairs
vidlens train scene --data data/captions.csv --epochs 10
```

See [docs/training.md](docs/training.md) for dataset formats, annotation tools, and training strategies.

---

## Web Interface

```bash
vidlens ui
# Opens at http://localhost:7860
```

Upload a video, select your lenses, and click Run — no command line required.

---

## Docker

```bash
# CPU
docker run -v $(pwd):/data ghcr.io/your-org/vidlens analyze /data/video.mp4 --lens objects

# GPU
docker run --gpus all -v $(pwd):/data ghcr.io/your-org/vidlens analyze /data/video.mp4 --lens objects
```

---

## Extending VidLens

Adding a new computer vision model takes around 50 lines. Subclass `BaseLens`, implement three methods, and register it — it immediately works in both the CLI and web UI.

```python
# vidlens/lenses/my_lens.py
from vidlens.lenses.base import BaseLens
import numpy as np

class MyCustomLens(BaseLens):
    name = "my_lens"
    description = "My custom analysis"

    def load_model(self):
        self.model = ...  # load your model here

    def process_frame(self, frame: np.ndarray) -> dict:
        result = self.model(frame)
        return {
            "lens": self.name,
            "detections": result,
            "annotated_frame": frame,
        }

    def train(self, data_config, epochs=50, **kwargs):
        ...  # optional: implement training support
```

Then register it in `vidlens/lenses/__init__.py`:

```python
from .my_lens import MyCustomLens
LENS_REGISTRY["my_lens"] = MyCustomLens
```

---

## Project Structure

```
vidlens/
├── vidlens/
│   ├── lenses/
│   │   ├── base.py                  # abstract base class for all lenses
│   │   ├── object_detection.py      # YOLOv8 with ByteTrack tracking
│   │   ├── face_detection.py        # YOLOv8-face with anonymization
│   │   ├── pose_estimation.py       # YOLOv8-pose, 17 keypoints
│   │   └── scene_classification.py  # CLIP zero-shot classification
│   ├── utils/
│   │   └── highlights.py            # CLIP-based highlight extraction
│   ├── ui/
│   │   └── app.py                   # Gradio web interface
│   ├── pipeline.py                  # core video processing loop
│   └── cli.py                       # CLI entry point
├── configs/
│   └── dataset_example.yaml         # dataset config template
├── docs/
│   ├── quickstart.md                # beginner setup guide
│   └── training.md                  # custom training guide
├── tests/
├── Dockerfile
├── pyproject.toml
└── setup.sh                         # one-command Linux installer
```

---

## Roadmap

- [x] Object detection and tracking (YOLOv8)
- [x] Face detection and anonymization
- [x] Pose estimation (17 keypoints)
- [x] Zero-shot scene classification (CLIP)
- [x] Highlight reel generation
- [x] Custom training for all lenses
- [x] Gradio web UI
- [ ] Depth estimation (MiDaS)
- [ ] OCR on video frames (EasyOCR)
- [ ] Optical flow (RAFT)
- [ ] Emotion recognition (DeepFace)
- [ ] Auto-subtitles (Whisper)
- [ ] Docker image with GPU support
- [ ] Plugin system for community lenses

---

## Contributing

Contributions are welcome. Pick an issue labeled `good first issue`, add a new lens, fix a bug, or improve the docs — then open a pull request.

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT — free for personal and commercial use.
