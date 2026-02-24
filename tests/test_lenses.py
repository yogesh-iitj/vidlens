"""
Basic tests for VidLens lenses.
Run with: pytest tests/
"""

import numpy as np
import pytest


def make_dummy_frame(h=480, w=640):
    """Generate a random BGR frame for testing."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestObjectDetectionLens:
    def test_init_default(self):
        from vidlens.lenses.object_detection import ObjectDetectionLens
        lens = ObjectDetectionLens()
        assert lens.name == "objects"
        assert lens.confidence == 0.25

    def test_init_custom(self):
        from vidlens.lenses.object_detection import ObjectDetectionLens
        lens = ObjectDetectionLens(variant="small", confidence=0.5, device="cpu")
        assert "yolov8s" in lens._weights

    def test_process_frame_returns_required_keys(self):
        """process_frame output must always have these keys."""
        from vidlens.lenses.object_detection import ObjectDetectionLens
        lens = ObjectDetectionLens(device="cpu")
        frame = make_dummy_frame()
        result = lens.process_frame(frame)
        assert "lens" in result
        assert "detections" in result
        assert "annotated_frame" in result
        assert isinstance(result["detections"], list)
        assert isinstance(result["annotated_frame"], np.ndarray)

    def test_train_raises_if_no_data(self):
        from vidlens.lenses.object_detection import ObjectDetectionLens
        lens = ObjectDetectionLens()
        with pytest.raises(Exception):
            lens.train("nonexistent_data.yaml", epochs=1)


class TestFaceLens:
    def test_anonymize_modes(self):
        from vidlens.lenses.face_detection import FaceLens
        for mode in ["blur", "pixelate", "black", "none"]:
            lens = FaceLens(anonymize=mode)
            assert lens.anonymize == mode

    def test_blur_strength_always_odd(self):
        from vidlens.lenses.face_detection import FaceLens
        lens = FaceLens(blur_strength=50)
        assert lens.blur_strength % 2 == 1


class TestPoseLens:
    def test_init(self):
        from vidlens.lenses.pose_estimation import PoseLens
        lens = PoseLens(variant="nano")
        assert "yolov8n-pose" in lens._weights


class TestSceneLens:
    def test_custom_labels(self):
        from vidlens.lenses.scene_classification import SceneClassificationLens
        labels = ["fire", "smoke", "normal"]
        lens = SceneClassificationLens(labels=labels)
        assert lens.labels == labels

    def test_top_k_capped(self):
        from vidlens.lenses.scene_classification import SceneClassificationLens
        lens = SceneClassificationLens(labels=["a", "b"], top_k=10)
        assert lens.top_k == 2  # capped at len(labels)


class TestPipeline:
    def test_unknown_lens_raises(self):
        from vidlens.pipeline import VideoPipeline
        import tempfile, os
        # Create a dummy video file path (doesn't need to exist for this test)
        with pytest.raises((ValueError, Exception)):
            p = VideoPipeline("fake.mp4", lenses=["not_a_real_lens"])

    def test_resolve_lens_from_string(self):
        from vidlens.pipeline import VideoPipeline
        from vidlens.lenses.object_detection import ObjectDetectionLens
        import tempfile
        p = VideoPipeline.__new__(VideoPipeline)
        p.lens_configs = {}
        resolved = p._resolve_lenses(["objects"])
        assert len(resolved) == 1
        assert isinstance(resolved[0], ObjectDetectionLens)


class TestLensRegistry:
    def test_all_lenses_registered(self):
        from vidlens.lenses import LENS_REGISTRY
        expected = {"objects", "faces", "pose", "scene"}
        assert expected.issubset(set(LENS_REGISTRY.keys()))

    def test_all_lenses_have_name_and_description(self):
        from vidlens.lenses import LENS_REGISTRY
        for name, cls in LENS_REGISTRY.items():
            assert hasattr(cls, "name")
            assert hasattr(cls, "description")
