"""
Highlight reel extraction using CLIP scene scores.
Picks the most "interesting" segments based on label similarity scores.
"""

from __future__ import annotations
from pathlib import Path
import json
import subprocess

import cv2
import numpy as np

from vidlens.lenses.scene_classification import SceneClassificationLens


def extract_highlights(
    video_path: str,
    lens: SceneClassificationLens,
    target_duration: int = 60,
    window_size: int = 5,
    output_dir: str = "output",
) -> str:
    """
    Score every window of frames with CLIP, select highest-scoring windows,
    stitch them into a highlight video using FFmpeg.

    Returns:
        Path to the output highlight video.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if lens.model is None:
        lens.load_model()

    print(f"Scoring {total_frames} frames with CLIP...")

    frame_scores: list[tuple[int, float]] = []  # (frame_idx, score)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = lens.process_frame(frame)
        score = result.get("top_score", 0.0)
        frame_scores.append((frame_idx, score))
        frame_idx += 1

    cap.release()

    # Smooth scores with a sliding window
    scores = np.array([s for _, s in frame_scores])
    kernel = np.ones(int(fps)) / int(fps)  # 1-second smoothing
    smoothed = np.convolve(scores, kernel, mode="same")

    # Select top non-overlapping windows
    frames_needed = int(target_duration * fps)
    window_frames = int(window_size * fps)
    selected_starts: list[int] = []
    temp = smoothed.copy()

    while sum(window_frames for _ in selected_starts) < frames_needed:
        best_start = int(np.argmax(temp))
        selected_starts.append(best_start)
        # Zero out this window to avoid overlap
        lo = max(0, best_start - window_frames)
        hi = min(len(temp), best_start + window_frames)
        temp[lo:hi] = 0
        if temp.max() < 0.01:
            break

    selected_starts.sort()

    # Write clip list for FFmpeg concat
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    segments_path = Path(output_dir) / "segments.txt"
    video_name = Path(video_path).stem

    with open(segments_path, "w") as f:
        for start in selected_starts:
            start_sec = start / fps
            end_sec = min((start + window_frames) / fps, total_frames / fps)
            # Extract each segment using FFmpeg
            seg_path = Path(output_dir) / f"seg_{start}.mp4"
            subprocess.run([
                "ffmpeg", "-y",
                "-ss", str(start_sec),
                "-to", str(end_sec),
                "-i", video_path,
                "-c", "copy",
                str(seg_path),
            ], capture_output=True)
            f.write(f"file '{seg_path.absolute()}'\n")

    # Concatenate segments
    out_path = Path(output_dir) / f"{video_name}_highlights.mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(segments_path),
        "-c", "copy",
        str(out_path),
    ], capture_output=True)

    # Save scores for analysis
    scores_path = Path(output_dir) / f"{video_name}_scores.json"
    with open(scores_path, "w") as f:
        json.dump(frame_scores, f)

    print(f"✅ Highlight reel saved: {out_path}")
    return str(out_path)
