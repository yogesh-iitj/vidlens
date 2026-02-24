"""
Lens registry — maps CLI names to lens classes.
Add your custom lens here to make it available in the CLI and UI.
"""

from .base import BaseLens
from .object_detection import ObjectDetectionLens
from .face_detection import FaceLens
from .pose_estimation import PoseLens
from .scene_classification import SceneClassificationLens

LENS_REGISTRY: dict[str, type[BaseLens]] = {
    "objects": ObjectDetectionLens,
    "faces":   FaceLens,
    "pose":    PoseLens,
    "scene":   SceneClassificationLens,
}

__all__ = [
    "BaseLens",
    "ObjectDetectionLens",
    "FaceLens",
    "PoseLens",
    "SceneClassificationLens",
    "LENS_REGISTRY",
]
