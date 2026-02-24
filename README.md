# 👁️ VidLens — Open Source Video Intelligence Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://python.org)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Run deep learning computer vision on any video — no training required. Custom training supported for all models.**

| Feature | Model | Pre-trained | Custom Training |
|---|---|---|---|
| Object Detection & Tracking | YOLOv8 | ✅ COCO 80 classes | ✅ Any classes |
| Face Detection & Anonymization | YOLOv8-face | ✅ | ✅ |
| Pose Estimation | YOLOv8-pose | ✅ 17 keypoints | ✅ |
| Scene Classification | CLIP | ✅ Zero-shot (any text labels) | ✅ |
| Highlight Reel Generation | CLIP | ✅ | — |

---

## 🚀 Installation

### ⭐ Easiest way (Linux — one command)

```bash
git clone https://github.com/your-org/vidlens
cd vidlens
bash setup.sh
```

That's it. The script handles Python, FFmpeg, GPU detection, model weights — everything.
👉 New to this? Read the [Beginner's Quickstart Guide](docs/quickstart.md)

### Manual install (all platforms)

```bash
pip install vidlens[all]         # everything
pip install vidlens[yolo]        # object/face/pose only
pip install vidlens[yolo,clip]   # + scene classification & highlights
```

---

## ⚡ Quick Start

### CLI

```bash
# Detect objects + track across frames
vidlens analyze myvideo.mp4 --lens objects

# Run multiple lenses at once
vidlens analyze myvideo.mp4 --lens objects --lens pose --lens scene

# Blur all faces for GDPR compliance
vidlens anonymize myvideo.mp4 --mode blur

# Generate a 60-second highlight reel
vidlens highlights match.mp4 --duration 60

# Generate highlights for specific events
vidlens highlights match.mp4 --labels "goal,tackle,celebration" --duration 90

# Train on your own data
vidlens train objects --data data/my_dataset.yaml --epochs 100

# Launch web UI
vidlens ui
```

### Python API

```python
from vidlens.pipeline import VideoPipeline

# Run analysis
pipeline = VideoPipeline(
    video_path="myvideo.mp4",
    lenses=["objects", "pose"],
    output_dir="output/",
)
summary = pipeline.run()
# → output/myvideo_annotated.mp4
# → output/myvideo_report.json
```

```python
# Use custom-trained model
from vidlens.lenses import ObjectDetectionLens

lens = ObjectDetectionLens(model_path="models/custom/best.pt")
```

```python
# Zero-shot scene detection — no training!
from vidlens.lenses import SceneClassificationLens

lens = SceneClassificationLens(labels=["fire", "smoke", "normal activity"])
```

---

## 🏋️ Custom Training

All lenses support fine-tuning on your own data:

```bash
# Fine-tune object detector on custom classes
vidlens train objects --data configs/dataset_example.yaml --epochs 100 --device cuda

# Fine-tune face detector for your domain
vidlens train faces --data data/faces.yaml --epochs 50

# Fine-tune CLIP on custom image-text pairs
vidlens train scene --data data/captions.csv --epochs 10
```

See [docs/training.md](docs/training.md) for full documentation on dataset formats and training strategies.

---

## 🌐 Web UI

```bash
vidlens ui
# → Opens at http://localhost:7860
```

Upload a video, select lenses, click Run. No command line needed.

---

## 🐳 Docker

```bash
# CPU
docker run -v $(pwd):/data ghcr.io/your-org/vidlens analyze /data/video.mp4 --lens objects

# GPU
docker run --gpus all -v $(pwd):/data ghcr.io/your-org/vidlens analyze /data/video.mp4 --lens objects
```

---

## 🔌 Add Your Own Lens

VidLens is designed to be extended. Adding a new computer vision model takes ~50 lines:

```python
# vidlens/lenses/my_lens.py
from vidlens.lenses.base import BaseLens
import numpy as np

class MyCustomLens(BaseLens):
    name = "my_lens"
    description = "My custom analysis"

    def load_model(self):
        # Load your model here
        self.model = ...

    def process_frame(self, frame: np.ndarray) -> dict:
        result = self.model(frame)
        return {
            "lens": self.name,
            "detections": result,
            "annotated_frame": frame,
        }

    def train(self, data_config, epochs=50, **kwargs):
        # Optional: implement training
        ...
```

Register it in `vidlens/lenses/__init__.py` and it immediately works with CLI + UI.

---

## 📁 Project Structure

```
vidlens/
├── vidlens/
│   ├── lenses/
│   │   ├── base.py              ← Abstract base class for all lenses
│   │   ├── object_detection.py  ← YOLOv8 objects + tracking
│   │   ├── face_detection.py    ← YOLOv8-face + anonymization
│   │   ├── pose_estimation.py   ← YOLOv8-pose (17 keypoints)
│   │   └── scene_classification.py ← CLIP zero-shot
│   ├── utils/
│   │   └── highlights.py        ← CLIP-based highlight extraction
│   ├── ui/
│   │   └── app.py               ← Gradio web interface
│   ├── pipeline.py              ← Core video processing loop
│   └── cli.py                   ← Typer CLI
├── configs/
│   └── dataset_example.yaml     ← Dataset config template
├── docs/
│   └── training.md              ← Training guide
├── tests/
├── pyproject.toml
└── README.md
```

---

## 🗺️ Roadmap

- [x] Object detection + tracking (YOLOv8)
- [x] Face detection + anonymization
- [x] Pose estimation (17 keypoints)
- [x] Zero-shot scene classification (CLIP)
- [x] Highlight reel generation
- [x] Custom training for all lenses
- [x] Gradio web UI
- [ ] Depth estimation (MiDaS)
- [ ] OCR on video frames (EasyOCR)
- [ ] Optical flow (RAFT)
- [ ] Emotion recognition (DeepFace)
- [ ] Crowd density estimation
- [ ] Auto-subtitles (Whisper)
- [ ] Docker image with GPU support
- [ ] Plugin system for community lenses

---

## 🤝 Contributing

1. Fork the repo
2. Pick an issue labeled `good first issue`
3. Add a new lens, fix a bug, or improve docs
4. Open a PR!

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

MIT — free for personal and commercial use.
