"""
Pose Estimation Lens — powered by YOLOv8-pose
Detects 17 human body keypoints per person. Supports custom training.
"""

from __future__ import annotations
from typing import Any
import numpy as np

from .base import BaseLens


# COCO 17-keypoint skeleton connections for drawing
SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # head
    (5, 6),                                   # shoulders
    (5, 7), (7, 9),                           # left arm
    (6, 8), (8, 10),                          # right arm
    (5, 11), (6, 12),                         # torso
    (11, 12),                                 # hips
    (11, 13), (13, 15),                       # left leg
    (12, 14), (14, 16),                       # right leg
]

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


class PoseLens(BaseLens):
    """
    Estimates human body pose (17 keypoints) for every person in frame.
    Great for sports analysis, fitness apps, gesture recognition.
    """

    name = "pose"
    description = "Human pose estimation using YOLOv8-pose (17 keypoints)"

    def __init__(
        self,
        model_path: str | None = None,
        variant: str = "nano",
        confidence: float = 0.3,
        keypoint_confidence: float = 0.5,
        device: str = "auto",
    ):
        """
        Args:
            variant: nano | small | medium | large
            confidence: Person detection threshold.
            keypoint_confidence: Min confidence to draw a keypoint.
        """
        super().__init__(model_path=model_path, device=device)
        self.confidence = confidence
        self.keypoint_confidence = keypoint_confidence

        weight_map = {
            "nano": "yolov8n-pose.pt",
            "small": "yolov8s-pose.pt",
            "medium": "yolov8m-pose.pt",
            "large": "yolov8l-pose.pt",
        }
        self._weights = model_path or weight_map.get(variant, "yolov8n-pose.pt")

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        from ultralytics import YOLO
        self.model = YOLO(self._weights)
        print(f"[PoseLens] Loaded: {self._weights} on {self.device}")

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        if self.model is None:
            self.load_model()

        results = self.model(
            frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )

        result = results[0]
        poses = []

        if result.keypoints is not None:
            for i, kp in enumerate(result.keypoints.data):
                keypoints = {}
                for j, (x, y, conf) in enumerate(kp.tolist()):
                    keypoints[KEYPOINT_NAMES[j]] = {
                        "x": x, "y": y, "confidence": conf,
                        "visible": conf >= self.keypoint_confidence,
                    }
                poses.append({
                    "person_id": i,
                    "keypoints": keypoints,
                    "bbox": result.boxes[i].xyxy[0].tolist() if result.boxes else None,
                })

        annotated = result.plot()

        return {
            "lens": self.name,
            "detections": poses,
            "count": len(poses),
            "annotated_frame": annotated,
        }

    # ------------------------------------------------------------------
    # Custom training
    # ------------------------------------------------------------------

    def train(
        self,
        data_config: str,
        epochs: int = 100,
        imgsz: int = 640,
        batch: int = 16,
        output_dir: str = "models/custom/pose",
        **kwargs,
    ) -> None:
        """
        Fine-tune on custom pose dataset.
        Use cases: custom keypoints (hands, animals, vehicles), domain-specific poses.

        Dataset format: YOLO pose format (COCO keypoint annotations converted to YOLO).
        See: docs/training.md#pose
        """
        from ultralytics import YOLO
        print(f"[PoseLens] Starting training on: {data_config}")
        model = YOLO(self._weights)
        results = model.train(
            data=data_config,
            task="pose",
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=output_dir,
            name="train",
            **kwargs,
        )
        print(f"✅ Training done. Best weights: {output_dir}/train/weights/best.pt")
        return results
