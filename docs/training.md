# VidLens — Custom Training Guide

Training custom models in VidLens is designed to be as simple as the inference.
All lenses share the same `train()` interface.

---

## 🎯 Quick Start

```bash
# Train object detector on custom classes
vidlens train objects --data configs/dataset_example.yaml --epochs 100

# Fine-tune face detector (e.g. for occlusion, masks, thermal)
vidlens train faces --data data/faces.yaml --epochs 50 --device cuda

# Fine-tune CLIP scene classifier on custom image-text pairs
vidlens train scene --data data/captions.csv --epochs 10
```

After training, use your model:
```bash
vidlens analyze video.mp4 --lens objects --model-path models/custom/objects/train/weights/best.pt
```

---

## 📁 Dataset Formats

### Object Detection & Face Detection (YOLO format)

```
my_dataset/
├── images/
│   ├── train/  ← your training images (.jpg or .png)
│   └── val/    ← validation images
└── labels/
    ├── train/  ← one .txt per image (same filename)
    └── val/
```

**Label file format** (`frame001.txt`):
```
# <class_id> <center_x> <center_y> <width> <height>  — all 0.0 to 1.0 normalized
0 0.512 0.433 0.210 0.385
1 0.231 0.710 0.105 0.201
```

**Dataset YAML** (`dataset.yaml`):
```yaml
path: my_dataset
train: images/train
val: images/val
names:
  0: cat
  1: dog
```

> 💡 **Recommended tool:** Use [Roboflow](https://roboflow.com) to annotate images and export directly in "YOLOv8" format.

---

### Pose Estimation (YOLO Pose format)

Same as object detection but labels include keypoint coordinates:

```
# <class_id> <cx> <cy> <w> <h> <kp1_x> <kp1_y> <kp1_vis> ... <kp17_x> <kp17_y> <kp17_vis>
0 0.5 0.5 0.4 0.8  0.48 0.12 2  0.52 0.14 2  ...
```

Visibility: `0`=not labeled, `1`=labeled but occluded, `2`=labeled and visible

---

### Scene Classification (CLIP fine-tuning)

A simple CSV file with image paths and text descriptions:

```csv
image_path,caption
data/images/frame001.jpg,"a person falling down stairs"
data/images/frame002.jpg,"crowd cheering at a sports event"
data/images/frame003.jpg,"empty parking lot at night"
```

---

## ⚙️ Training Options

| Option | Default | Description |
|---|---|---|
| `--epochs` | 50 | Training epochs |
| `--batch` | 16 | Batch size (-1 = auto) |
| `--device` | auto | cpu / cuda / mps |
| `--model-path` | pretrained | Start from custom checkpoint |
| `--output-dir` | models/custom | Where to save weights |

---

## 🔁 Using Trained Models

**In CLI:**
```bash
vidlens analyze video.mp4 --lens objects
# Add --model-path flag (coming in v0.2)
```

**In Python:**
```python
from vidlens.lenses import ObjectDetectionLens

lens = ObjectDetectionLens(model_path="models/custom/objects/train/weights/best.pt")
```

---

## 🧠 Pre-trained → Custom: Recommended Strategy

| Scenario | Strategy |
|---|---|
| New classes, similar domain | Fine-tune for 50–100 epochs |
| Very different domain (medical, satellite) | Fine-tune for 200+ epochs, lower LR |
| Very small dataset (<50 images/class) | Use `nano` or `small` variant |
| Large dataset, accuracy critical | Use `large` or `xlarge` variant |

---

## 📦 Exporting Your Model

```python
from vidlens.lenses import ObjectDetectionLens

lens = ObjectDetectionLens(model_path="best.pt")
lens.export("model.onnx", format="onnx")       # ONNX (cross-platform)
lens.export("model.torchscript", format="torchscript")  # TorchScript
```

---

## 🔌 Adding a New Lens

1. Create `vidlens/lenses/my_lens.py` subclassing `BaseLens`
2. Implement `load_model()`, `process_frame()`, and `train()`
3. Register in `vidlens/lenses/__init__.py`:
   ```python
   from .my_lens import MyLens
   LENS_REGISTRY["my_lens"] = MyLens
   ```
4. Done! Now `vidlens analyze video.mp4 --lens my_lens` works.
