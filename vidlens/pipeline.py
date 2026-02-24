"""
Video Pipeline — reads a video, runs selected lenses frame by frame,
writes annotated output video and JSON report.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .lenses.base import BaseLens
from .lenses import LENS_REGISTRY


class VideoPipeline:
    """
    Orchestrates running one or more lenses over a video file.

    Example:
        pipeline = VideoPipeline("input.mp4", lenses=["objects", "pose"])
        pipeline.run(output_dir="output/")
    """

    def __init__(
        self,
        video_path: str,
        lenses: list[str | BaseLens],
        output_dir: str = "output",
        show_preview: bool = False,
        save_json: bool = True,
        save_video: bool = True,
        frame_skip: int = 0,
        lens_configs: dict | None = None,
    ):
        """
        Args:
            video_path: Path to input video file.
            lenses: List of lens names (str) or instantiated BaseLens objects.
            output_dir: Where to save annotated video and JSON report.
            show_preview: Show live preview window while processing.
            save_json: Save per-frame JSON report.
            save_video: Save annotated output video.
            frame_skip: Process every Nth frame (0 = all frames).
            lens_configs: Dict of lens_name → kwargs for lens construction.
        """
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.show_preview = show_preview
        self.save_json = save_json
        self.save_video = save_video
        self.frame_skip = frame_skip
        self.lens_configs = lens_configs or {}

        # Resolve lenses
        self.lenses: list[BaseLens] = self._resolve_lenses(lenses)

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, on_progress: Callable[[int, int], None] | None = None) -> dict:
        """
        Process the video through all lenses.

        Args:
            on_progress: Optional callback(current_frame, total_frames).

        Returns:
            Summary dict with stats and output file paths.
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        # Video metadata
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"\n📹 Video   : {self.video_path.name}")
        print(f"   Frames  : {total_frames} @ {fps:.1f} fps")
        print(f"   Size    : {width}×{height}")
        print(f"   Lenses  : {[l.name for l in self.lenses]}\n")

        # Output video writer
        out_video_path = self.output_dir / f"{self.video_path.stem}_annotated.mp4"
        writer = None
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (width, height))

        # Pre-load all models
        for lens in self.lenses:
            if lens.model is None:
                lens.load_model()

        report = {
            "video": str(self.video_path),
            "fps": fps,
            "total_frames": total_frames,
            "lenses": [l.name for l in self.lenses],
            "frames": [],
        }

        frame_idx = 0
        processed = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Frame skipping
            if self.frame_skip > 0 and frame_idx % (self.frame_skip + 1) != 0:
                frame_idx += 1
                continue

            frame_result = {
                "frame": frame_idx,
                "timestamp_sec": round(frame_idx / fps, 3),
                "results": {},
            }

            # Run all lenses, overlay results onto frame
            annotated = frame.copy()
            for lens in self.lenses:
                result = lens.process_frame(frame)
                # Use last lens's annotated frame for display
                annotated = result.pop("annotated_frame", annotated)
                frame_result["results"][lens.name] = result

            report["frames"].append(frame_result)

            if writer:
                writer.write(annotated)

            if self.show_preview:
                cv2.imshow("VidLens Preview", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Preview closed by user.")
                    break

            processed += 1
            frame_idx += 1

            if on_progress:
                on_progress(frame_idx, total_frames)

            # Print progress every 100 frames
            if processed % 100 == 0:
                elapsed = time.time() - start_time
                fps_actual = processed / elapsed
                remaining = (total_frames - frame_idx) / max(fps_actual, 1)
                print(f"  Frame {frame_idx}/{total_frames} | {fps_actual:.1f} fps | ETA: {remaining:.0f}s")

        cap.release()
        if writer:
            writer.release()
        if self.show_preview:
            cv2.destroyAllWindows()

        # Save JSON report
        out_json_path = None
        if self.save_json:
            out_json_path = self.output_dir / f"{self.video_path.stem}_report.json"
            with open(out_json_path, "w") as f:
                json.dump(report, f, indent=2)

        elapsed = time.time() - start_time
        summary = {
            "processed_frames": processed,
            "elapsed_seconds": round(elapsed, 2),
            "avg_fps": round(processed / elapsed, 2),
            "output_video": str(out_video_path) if self.save_video else None,
            "output_json": str(out_json_path) if out_json_path else None,
        }

        print(f"\n✅ Done! Processed {processed} frames in {elapsed:.1f}s ({summary['avg_fps']} fps)")
        if self.save_video:
            print(f"   Video  → {out_video_path}")
        if out_json_path:
            print(f"   Report → {out_json_path}")

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_lenses(self, lenses: list[str | BaseLens]) -> list[BaseLens]:
        resolved = []
        for lens in lenses:
            if isinstance(lens, BaseLens):
                resolved.append(lens)
            elif isinstance(lens, str):
                if lens not in LENS_REGISTRY:
                    available = list(LENS_REGISTRY.keys())
                    raise ValueError(f"Unknown lens '{lens}'. Available: {available}")
                cfg = self.lens_configs.get(lens, {})
                resolved.append(LENS_REGISTRY[lens](**cfg))
            else:
                raise TypeError(f"Lens must be a string name or BaseLens instance, got {type(lens)}")
        return resolved
