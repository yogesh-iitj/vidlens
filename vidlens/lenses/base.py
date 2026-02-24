"""
Base class for all VidLens analysis modules (lenses).
Every lens — pre-trained or custom-trained — inherits from this.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import numpy as np


class BaseLens(ABC):
    """
    Abstract base for every VidLens lens.

    To add a new lens:
      1. Subclass BaseLens
      2. Implement load_model(), process_frame(), and (optionally) train()
      3. Register it in vidlens/lenses/__init__.py
    """

    # Human-readable name shown in CLI / UI
    name: str = "base"
    description: str = "Base lens"

    def __init__(self, model_path: str | None = None, device: str = "auto", **kwargs):
        """
        Args:
            model_path: Path to custom weights. If None, pre-trained weights are used.
            device: 'cpu', 'cuda', 'mps', or 'auto' (auto-detects best device).
        """
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self.model = None
        self.config = kwargs

    # ------------------------------------------------------------------
    # Core interface — every lens MUST implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights into self.model."""
        ...

    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Run inference on a single BGR frame (OpenCV format).

        Returns a dict with at minimum:
            {
                "lens": self.name,
                "detections": [...],   # list of result dicts
                "annotated_frame": np.ndarray  # frame with overlays drawn
            }
        """
        ...

    # ------------------------------------------------------------------
    # Optional interface — implement for custom training support
    # ------------------------------------------------------------------

    def train(self, data_config: str, epochs: int = 50, **kwargs) -> None:
        """
        Fine-tune or train from scratch on custom data.
        Override in subclasses that support training.

        Args:
            data_config: Path to dataset config YAML (YOLO format or custom).
            epochs: Number of training epochs.
        """
        raise NotImplementedError(
            f"Lens '{self.name}' does not yet support custom training. "
            "Contribute training support at github.com/your-org/vidlens!"
        )

    def export(self, output_path: str, format: str = "onnx") -> None:
        """
        Export trained model to a portable format (ONNX, TorchScript, etc.)
        Override in subclasses.
        """
        raise NotImplementedError(f"Export not implemented for lens '{self.name}'.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def __repr__(self):
        return f"<Lens: {self.name} | device={self.device} | model={self.model_path or 'pretrained'}>"
