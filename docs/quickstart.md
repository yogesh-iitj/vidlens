# 🚀 VidLens — Beginner's Quickstart Guide

No prior deep learning experience needed. This guide walks you through everything step by step.

---

## Step 1: Download the project

Open a terminal and run:

```bash
git clone https://github.com/your-org/vidlens.git
cd vidlens
```

> **Don't have git?** Run: `sudo apt install git`

---

## Step 2: Run the setup script

This script does everything for you automatically:
- Checks your Python version
- Installs FFmpeg and other tools
- Creates an isolated Python environment
- Detects your GPU (if you have one)
- Installs all dependencies
- Downloads model weights

```bash
bash setup.sh
```

⏱ Takes about 2–5 minutes the first time.

---

## Step 3: Activate the environment

Every time you open a new terminal, run this first:

```bash
source activate.sh
```

You'll see `(venv)` appear in your terminal prompt — that means it's working.

---

## Step 4: Try it out!

### Analyze a video (detect objects)
```bash
vidlens analyze myvideo.mp4 --lens objects
```

Results saved to `output/myvideo_annotated.mp4` ✅

### Detect multiple things at once
```bash
vidlens analyze myvideo.mp4 --lens objects --lens pose
```

### Blur all faces (great for sharing videos publicly)
```bash
vidlens anonymize myvideo.mp4 --mode blur
```

### Use the web interface instead (no commands needed!)
```bash
vidlens ui
```
Then open your browser at **http://localhost:7860** — upload your video, click Run!

---

## What does each lens do?

| Lens | What it detects | Command |
|---|---|---|
| `objects` | 80 types of objects (people, cars, dogs...) | `--lens objects` |
| `faces` | Human faces, with optional blurring | `--lens faces` |
| `pose` | Body position and keypoints (arms, legs...) | `--lens pose` |
| `scene` | What type of scene/activity is happening | `--lens scene` |

---

## Where do results go?

Everything is saved to the `output/` folder:

```
output/
├── myvideo_annotated.mp4   ← video with boxes/overlays drawn
└── myvideo_report.json     ← frame-by-frame detection data
```

---

## Running slower on CPU? Try these tips:

```bash
# Skip every other frame (2x faster, still good results)
vidlens analyze myvideo.mp4 --lens objects --frame-skip 1

# Skip 2 frames between each processed frame (3x faster)
vidlens analyze myvideo.mp4 --lens objects --frame-skip 2
```

---

## Train on your own data (when you're ready)

Once you have custom labeled data, training is one command:

```bash
vidlens train objects --data my_data/dataset.yaml --epochs 50
```

See [training.md](training.md) for how to label your own video frames.

---

## Something not working?

**"command not found: vidlens"**
→ Run `source activate.sh` first

**Video processing is very slow**
→ Add `--frame-skip 2` to your command

**"ModuleNotFoundError"**
→ Re-run `bash setup.sh`

**GPU not being used**
→ Run `python3 -c "import torch; print(torch.cuda.is_available())"` — should print `True`

---

## Get help

```bash
vidlens --help              # overview of all commands
vidlens analyze --help      # options for the analyze command
vidlens train --help        # options for training
vidlens lenses              # list all available lenses
```
