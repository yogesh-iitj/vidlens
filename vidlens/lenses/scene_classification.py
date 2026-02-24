"""
Scene Classification Lens — powered by OpenAI CLIP
Zero-shot classification: describe any category in plain English, no re-training needed.
Also supports fine-tuning CLIP on custom image-text pairs.
"""

from __future__ import annotations
from typing import Any
import numpy as np

from .base import BaseLens


DEFAULT_LABELS = [
    "indoor scene", "outdoor scene",
    "sports activity", "crowd of people", "empty space",
    "vehicle traffic", "nature landscape", "urban street",
    "office environment", "construction site",
]


class SceneClassificationLens(BaseLens):
    """
    Zero-shot scene/activity classification using CLIP.

    No training needed — just describe what you want to detect in plain English.
    The model scores each frame against your custom labels.

    Example:
        lens = SceneClassificationLens(labels=["fire", "smoke", "normal"])
        # Now it classifies every frame as fire/smoke/normal — no training!
    """

    name = "scene"
    description = "Zero-shot scene classification using CLIP"

    def __init__(
        self,
        model_path: str | None = None,
        labels: list[str] | None = None,
        model_name: str = "ViT-B/32",
        top_k: int = 3,
        device: str = "auto",
        sample_every_n_frames: int = 15,  # CLIP is heavier, skip frames
    ):
        """
        Args:
            labels: Text labels to classify against. Uses DEFAULT_LABELS if None.
            model_name: CLIP variant. Options: ViT-B/32, ViT-B/16, ViT-L/14
            top_k: Return top-k label predictions per frame.
            sample_every_n_frames: Only run CLIP every N frames (performance).
        """
        super().__init__(model_path=model_path, device=device)
        self.labels = labels or DEFAULT_LABELS
        self.model_name = model_name
        self.top_k = min(top_k, len(self.labels))
        self.sample_every_n = sample_every_n_frames

        self._text_features = None
        self._frame_count = 0
        self._last_result: dict | None = None

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        import clip
        import torch
        self.model, self.preprocess = clip.load(self.model_name, device=self.device)
        self._encode_labels()
        print(f"[SceneLens] Loaded CLIP {self.model_name} | {len(self.labels)} labels")

    def _encode_labels(self) -> None:
        """Pre-encode text labels (only done once)."""
        import clip
        import torch
        tokens = clip.tokenize(self.labels).to(self.device)
        with torch.no_grad():
            self._text_features = self.model.encode_text(tokens)
            self._text_features /= self._text_features.norm(dim=-1, keepdim=True)

    def set_labels(self, labels: list[str]) -> None:
        """Update labels at runtime without reloading the model."""
        self.labels = labels
        self.top_k = min(self.top_k, len(labels))
        if self.model is not None:
            self._encode_labels()

    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        if self.model is None:
            self.load_model()

        self._frame_count += 1

        # Skip frames for performance — return last result on skipped frames
        if self._frame_count % self.sample_every_n != 0 and self._last_result is not None:
            return {**self._last_result, "annotated_frame": self._draw_overlay(frame, self._last_result)}

        import clip
        import torch
        from PIL import Image

        # Convert BGR → PIL
        image = Image.fromarray(frame[:, :, ::-1])
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarities = (100.0 * image_features @ self._text_features.T).softmax(dim=-1)

        scores = similarities[0].cpu().tolist()
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.top_k]

        predictions = [
            {"label": self.labels[i], "score": round(scores[i], 4)}
            for i in top_indices
        ]

        result = {
            "lens": self.name,
            "detections": predictions,
            "top_label": predictions[0]["label"],
            "top_score": predictions[0]["score"],
        }
        self._last_result = result

        return {**result, "annotated_frame": self._draw_overlay(frame, result)}

    def _draw_overlay(self, frame: np.ndarray, result: dict) -> np.ndarray:
        import cv2
        out = frame.copy()
        y = 30
        for pred in result.get("detections", []):
            text = f"{pred['label']}: {pred['score']:.1%}"
            cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            y += 28
        return out

    # ------------------------------------------------------------------
    # Custom training (CLIP fine-tuning)
    # ------------------------------------------------------------------

    def train(
        self,
        data_config: str,
        epochs: int = 10,
        lr: float = 1e-5,
        batch: int = 32,
        output_dir: str = "models/custom/scene",
        **kwargs,
    ) -> None:
        """
        Fine-tune CLIP on custom image-text pairs.

        data_config: Path to a CSV with columns: image_path, caption
                     e.g.:  data/images/frame001.jpg, "a person falling down stairs"

        This updates both image and text encoders for domain adaptation
        (e.g., medical video, satellite imagery, industrial inspection).
        """
        import torch
        import clip
        from torch.utils.data import DataLoader
        from pathlib import Path
        import pandas as pd
        from PIL import Image

        print(f"[SceneLens] Fine-tuning CLIP on: {data_config}")
        df = pd.read_csv(data_config)

        model, preprocess = clip.load(self.model_name, device=self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_loss = float("inf")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            total_loss = 0
            for _, row in df.iterrows():
                image = preprocess(Image.open(row["image_path"])).unsqueeze(0).to(self.device)
                text = clip.tokenize([row["caption"]]).to(self.device)

                logits_per_image, logits_per_text = model(image, text)
                labels = torch.arange(len(image)).to(self.device)
                loss = (
                    torch.nn.functional.cross_entropy(logits_per_image, labels) +
                    torch.nn.functional.cross_entropy(logits_per_text, labels)
                ) / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(df)
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = Path(output_dir) / "best_clip.pt"
                torch.save(model.state_dict(), save_path)

        print(f"✅ CLIP fine-tuning complete. Best model: {output_dir}/best_clip.pt")
