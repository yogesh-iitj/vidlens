"""
Object Detection Lens — powered by YOLOv8 (Ultralytics)
Pre-trained on COCO (80 classes). Supports custom training.
"""

from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np

from .base import BaseLens


# Default pre-trained weight options
PRETRAINED_WEIGHTS = {
    "nano":   "yolov8n.pt",   # fastest, least accurate
    "small":  "yolov8s.pt",
    "medium": "yolov8m.pt",
    "large":  "yolov8l.pt",
    "xlarge": "yolov8x.pt",   # slowest, most accurate
}


class ObjectDetectionLens(BaseLens):
    """
    Detects and tracks 80 COCO object classes in every frame.
    Can be fine-tuned on custom classes via the train() method.
    """

    name = "objects"
    description = "Detect & track objects using YOLOv8 (COCO pre-trained)"

    def __init__(
        self,
        model_path: str | None = None,
        variant: str = "nano",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
        track: bool = True,
    ):
        """
        Args:
            model_path: Path to custom .pt weights. If None, downloads pre-trained.
            variant: Pre-trained size: nano | small | medium | large | xlarge
            confidence: Detection confidence threshold (0–1).
            iou_threshold: NMS IoU threshold.
            track: If True, uses ByteTrack for object ID tracking across frames.
        """
        super().__init__(model_path=model_path, device=device)
        self.variant = variant
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.track = track

        self._weights = model_path or PRETRAINED_WEIGHTS[variant]

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """Downloads pre-trained weights on first run, caches locally."""
        from ultralytics import YOLO
        self.model = YOLO(self._weights)
        print(f"[ObjectDetectionLens] Loaded: {self._weights} on {self.device}")

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Args:
            frame: BGR numpy array from OpenCV.
        Returns:
            {
                "lens": "objects",
                "detections": [{"id", "label", "confidence", "bbox": [x1,y1,x2,y2]}, ...],
                "annotated_frame": np.ndarray
            }
        """
        if self.model is None:
            self.load_model()

        if self.track:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        else:
            results = self.model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )

        detections = []
        result = results[0]

        for i, box in enumerate(result.boxes):
            det = {
                "label": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
            }
            # Track ID is available when tracking is enabled
            if self.track and box.id is not None:
                det["id"] = int(box.id)
            detections.append(det)

        annotated = result.plot()  # Returns BGR frame with drawn boxes

        return {
            "lens": self.name,
            "detections": detections,
            "count": len(detections),
            "annotated_frame": annotated,
        }

    # ------------------------------------------------------------------
    # Custom training support
    # ------------------------------------------------------------------

    def train(
        self,
        data_config: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        output_dir: str = "models/custom/objects",
        resume: bool = False,
        **kwargs,
    ) -> None:
        """
        Fine-tune on custom data.

        Args:
            data_config: Path to dataset YAML (Ultralytics YOLO format).
                         See docs/training.md for format details.
            epochs: Training epochs.
            imgsz: Input image size.
            batch: Batch size (-1 for auto).
            output_dir: Where to save trained weights.
            resume: Resume from last checkpoint if True.

        Example:
            lens = ObjectDetectionLens(variant="small")
            lens.train("data/my_dataset.yaml", epochs=100)

        Dataset YAML format:
            path: data/my_dataset
            train: images/train
            val: images/val
            names:
              0: cat
              1: dog
        """
        from ultralytics import YOLO

        print(f"[ObjectDetectionLens] Starting training on: {data_config}")
        print(f"  Base weights : {self._weights}")
        print(f"  Epochs       : {epochs}")
        print(f"  Device       : {self.device}")

        model = YOLO(self._weights)
        results = model.train(
            data=data_config,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=output_dir,
            name="train",
            resume=resume,
            **kwargs,
        )

        best_weights = Path(output_dir) / "train" / "weights" / "best.pt"
        print(f"\n✅ Training complete! Best weights saved to: {best_weights}")
        print("   Load with: ObjectDetectionLens(model_path='path/to/best.pt')")
        return results

    def export(self, output_path: str = "model.onnx", format: str = "onnx") -> None:
        """Export model to ONNX or TorchScript for deployment."""
        if self.model is None:
            self.load_model()
        self.model.export(format=format, output=output_path)
        print(f"[ObjectDetectionLens] Exported to {format}: {output_path}")
