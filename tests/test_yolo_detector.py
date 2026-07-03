from pathlib import Path

import numpy as np
import pytest

import visual_deploy.detection.yolo_detector as yolo_detector
from visual_deploy.detection.yolo_detector import MockDetector, UltralyticsYoloDetector


def test_mock_detector_returns_center_box():
    det = MockDetector(confidence=0.8)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = det.infer(image)
    assert len(out) == 1
    assert out[0].bbox_xyxy == (50.0, 25.0, 150.0, 75.0)
    assert out[0].confidence == 0.8


@pytest.mark.parametrize("shape", [(100, 200), (100, 200, 1), (100, 200, 4)])
def test_mock_detector_rejects_non_bgr_images(shape):
    det = MockDetector()
    image = np.zeros(shape, dtype=np.uint8)

    with pytest.raises(ValueError, match="color_bgr"):
        det.infer(image)


def test_ultralytics_detector_rejects_missing_weights(tmp_path):
    missing = tmp_path / "missing.pt"

    with pytest.raises(FileNotFoundError, match="missing.pt"):
        UltralyticsYoloDetector(missing)


def test_ultralytics_detector_requires_optional_dependency(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"placeholder")
    monkeypatch.setattr(yolo_detector, "YOLO", None)
    monkeypatch.setattr(yolo_detector, "YOLOv10", None)
    monkeypatch.setattr(yolo_detector, "_YOLO_IMPORT_ERROR", RuntimeError("broken ultralytics install"))

    with pytest.raises(ImportError, match="broken ultralytics install"):
        UltralyticsYoloDetector(weights)


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1, float("nan"), float("inf"), None, "bad"])
def test_ultralytics_detector_rejects_invalid_conf_threshold(tmp_path, monkeypatch, bad_threshold):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"placeholder")

    class FakeModel:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(yolo_detector, "YOLO", FakeModel)

    with pytest.raises(ValueError, match="conf_threshold"):
        UltralyticsYoloDetector(weights, conf_threshold=bad_threshold)


def test_ultralytics_detector_parses_first_result_boxes(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"placeholder")

    class FakeScalar:
        def __init__(self, value):
            self.value = value

        def item(self):
            return self.value

    class FakeArray:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    class FakeBox:
        xyxy = [FakeArray([1.0, 2.0, 3.0, 4.0])]
        conf = [FakeScalar(0.7)]
        cls = [FakeScalar(2)]

    class FakeResult:
        boxes = [FakeBox()]
        names = {2: "apple"}

    class FakeModel:
        def __init__(self, path):
            self.path = path
            self.calls = []

        def predict(self, **kwargs):
            self.calls.append(kwargs)
            return [FakeResult()]

    monkeypatch.setattr(yolo_detector, "YOLO", FakeModel)
    monkeypatch.setattr(yolo_detector, "YOLOv10", None)
    image = np.zeros((10, 20, 3), dtype=np.uint8)

    det = UltralyticsYoloDetector(weights, conf_threshold=0.4, device="cpu", model_type="yolo")
    out = det.infer(image)

    assert det.model.path == str(weights)
    assert det.model.calls == [{"source": image, "conf": 0.4, "verbose": False, "device": "cpu"}]
    assert len(out) == 1
    assert out[0].bbox_xyxy == (1.0, 2.0, 3.0, 4.0)
    assert out[0].confidence == 0.7
    assert out[0].class_id == 2
    assert out[0].label == "apple"


def test_ultralytics_detector_handles_empty_boxes(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"placeholder")

    class FakeResult:
        boxes = None
        names = {}

    class FakeModel:
        def __init__(self, path):
            self.path = path

        def predict(self, **kwargs):
            return [FakeResult()]

    monkeypatch.setattr(yolo_detector, "YOLO", FakeModel)
    monkeypatch.setattr(yolo_detector, "YOLOv10", None)
    det = UltralyticsYoloDetector(weights, model_type="yolo")

    assert det.infer(np.zeros((10, 20, 3), dtype=np.uint8)) == []


def test_ultralytics_detector_can_select_yolov10_model_class(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"placeholder")

    class FakeYoloV10:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(yolo_detector, "YOLOv10", FakeYoloV10)

    det = UltralyticsYoloDetector(weights, model_type="yolov10")

    assert isinstance(det.model, FakeYoloV10)
    assert det.model_type == "yolov10"


def test_ultralytics_detector_rejects_missing_yolov10_class(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"placeholder")
    monkeypatch.setattr(yolo_detector, "YOLOv10", None)

    with pytest.raises(ImportError, match="YOLOv10"):
        UltralyticsYoloDetector(weights, model_type="yolov10")


def test_ultralytics_detector_rejects_unknown_model_type(tmp_path, monkeypatch):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="model_type"):
        UltralyticsYoloDetector(weights, model_type="bad")


def test_ultralytics_detector_weight_smoke_if_available():
    weights = Path("weights/yolo_detect.pt")
    if not weights.exists():
        pytest.skip("weights/yolo_detect.pt not present")
    if yolo_detector.YOLO is None:
        pytest.skip(f"ultralytics is unavailable: {yolo_detector._YOLO_IMPORT_ERROR}")

    det = UltralyticsYoloDetector(weights, device="cpu", model_type="auto")
    out = det.infer(np.zeros((64, 64, 3), dtype=np.uint8))

    assert isinstance(out, list)
