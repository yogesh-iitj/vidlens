"""
Face Detection & Anonymization Lens — powered by YOLOv8-face / MediaPipe
Detects faces, optionally blurs or pixelates them for privacy compliance.
Supports custom fine-tuning on domain-specific face data.
"""

from __future__ import annotations
from typing import Any, Literal
import numpy as np
import cv2

from .base import BaseLens


BlurMode = Literal["blur", "pixelate", "black", "none"]


class FaceLens(BaseLens):
    """
    Detects faces in every frame.
    Privacy mode can blur, pixelate, or blackout all detected faces.
    Useful for GDPR compliance, anonymizing footage before sharing.
    """

    name = "faces"
    description = "Detect & anonymize faces using YOLOv8-face"

    def __init__(
        self,
        model_path: str | None = None,
        confidence: float = 0.4,
        anonymize: BlurMode = "none",
        blur_strength: int = 51,
        device: str = "auto",
    ):
        """
        Args:
            model_path: Custom weights path. Defaults to yolov8n-face.pt.
            confidence: Detection threshold.
            anonymize: What to do with detected faces:
                       'blur'      → Gaussian blur
                       'pixelate' → pixelation effect
                       'black'    → solid black rectangle
                       'none'     → just draw bounding boxes
            blur_strength: Blur kernel size (must be odd). Higher = more blurred.
        """
        super().__init__(model_path=model_path, device=device)
        self.confidence = confidence
        self.anonymize = anonymize
        self.blur_strength = blur_strength if blur_strength % 2 == 1 else blur_strength + 1

        # YOLOv8-face pre-trained weights (auto-downloaded)
        self._weights = model_path or "yolov8n-face.pt"

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        from ultralytics import YOLO
        self.model = YOLO(self._weights)
        print(f"[FaceLens] Loaded: {self._weights} on {self.device}")

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
        faces = []
        output_frame = frame.copy()

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf)

            faces.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": conf,
            })

            # Apply anonymization
            if self.anonymize != "none":
                output_frame = self._anonymize_region(output_frame, x1, y1, x2, y2)
            else:
                # Just draw a box
                cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    output_frame, f"face {conf:.2f}",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )

        return {
            "lens": self.name,
            "detections": faces,
            "count": len(faces),
            "anonymize_mode": self.anonymize,
            "annotated_frame": output_frame,
        }

    # ------------------------------------------------------------------
    # Custom training
    # ------------------------------------------------------------------

    def train(
        self,
        data_config: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        output_dir: str = "models/custom/faces",
        **kwargs,
    ) -> None:
        """
        Fine-tune face detector on custom data (e.g. specific demographics,
        occlusion types, thermal images, low-light footage).

        Dataset format: same as YOLO — one class: 0: face
        """
        from ultralytics import YOLO

        print(f"[FaceLens] Starting fine-tuning on: {data_config}")
        model = YOLO(self._weights)
        results = model.train(
            data=data_config,
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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _anonymize_region(
        self, frame: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> np.ndarray:
        """Apply the selected anonymization method to a bounding box region."""
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame

        if self.anonymize == "blur":
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(
                roi, (self.blur_strength, self.blur_strength), 0
            )
        elif self.anonymize == "pixelate":
            h, w = roi.shape[:2]
            small = cv2.resize(roi, (max(1, w // 10), max(1, h // 10)))
            frame[y1:y2, x1:x2] = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        elif self.anonymize == "black":
            frame[y1:y2, x1:x2] = 0

        return frame
