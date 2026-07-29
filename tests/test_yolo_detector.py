import numpy as np
import pytest

from visual_deploy.detection.yolo_detector import MockDetector, YoloV10Detector


class FakeYoloSession:
    input_names = ("images",)
    output_names = ("output0",)

    def __init__(self, output=None):
        self.output = np.zeros((1, 300, 6), dtype=np.float32) if output is None else output
        self.inputs = None

    def run(self, inputs):
        self.inputs = inputs
        return {"output0": self.output}


def test_mock_detector_returns_center_box():
    out = MockDetector(confidence=0.8).infer(np.zeros((100, 200, 3), dtype=np.uint8))
    assert out[0].bbox_xyxy == (50.0, 25.0, 150.0, 75.0)
    assert out[0].confidence == 0.8


@pytest.mark.parametrize("shape", [(100, 200), (100, 200, 1), (100, 200, 4)])
def test_mock_detector_rejects_non_bgr_images(shape):
    with pytest.raises(ValueError, match="color_bgr"):
        MockDetector().infer(np.zeros(shape, dtype=np.uint8))


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.1, float("nan"), float("inf"), None, "bad"])
def test_yolov10_rejects_invalid_conf_threshold(bad_threshold):
    with pytest.raises(ValueError, match="conf_threshold"):
        YoloV10Detector("unused.onnx", conf_threshold=bad_threshold, session=FakeYoloSession())


def test_yolov10_preprocesses_bgr_and_maps_letterbox_boxes():
    output = np.zeros((1, 300, 6), dtype=np.float32)
    # Original 200x100 image is resized to 640x320 and padded 160 px vertically.
    output[0, 0] = [160.0, 240.0, 480.0, 400.0, 0.9, 0.0]
    session = FakeYoloSession(output)
    detector = YoloV10Detector("unused.onnx", conf_threshold=0.5, session=session)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image[:, :, 0] = 255

    detections = detector.infer(image)

    assert session.inputs["images"].shape == (1, 3, 640, 640)
    assert session.inputs["images"][0, 2, 200, 320] == pytest.approx(1.0)  # BGR blue -> RGB channel 2
    assert detections[0].bbox_xyxy == pytest.approx((50.0, 25.0, 150.0, 75.0))
    assert detections[0].label == "apple"


def test_yolov10_filters_low_scores_and_invalid_boxes():
    output = np.zeros((1, 300, 6), dtype=np.float32)
    output[0, 0] = [10, 10, 20, 20, 0.49, 0]
    output[0, 1] = [20, 20, 10, 10, 0.9, 0]
    detector = YoloV10Detector("unused.onnx", conf_threshold=0.5, session=FakeYoloSession(output))
    assert detector.infer(np.zeros((640, 640, 3), dtype=np.uint8)) == []
    assert detector.last_diagnostics.output_candidate_count == 300
    assert detector.last_diagnostics.above_threshold_count == 1
    assert detector.last_diagnostics.max_confidence == pytest.approx(0.9)


def test_yolov10_diagnostics_distinguish_below_threshold_output():
    output = np.zeros((1, 3, 6), dtype=np.float32)
    output[0, :, 4] = [0.1, 0.3, 0.49]
    detector = YoloV10Detector("unused.onnx", conf_threshold=0.5, session=FakeYoloSession(output))

    assert detector.infer(np.zeros((640, 640, 3), dtype=np.uint8)) == []
    assert detector.last_diagnostics.output_candidate_count == 3
    assert detector.last_diagnostics.finite_confidence_count == 3
    assert detector.last_diagnostics.above_threshold_count == 0
    assert detector.last_diagnostics.max_confidence == pytest.approx(0.49)


def test_yolov10_rejects_unexpected_output_shape():
    session = FakeYoloSession(np.zeros((1, 84, 8400), dtype=np.float32))
    detector = YoloV10Detector("unused.onnx", session=session)
    with pytest.raises(ValueError, match="output"):
        detector.infer(np.zeros((640, 640, 3), dtype=np.uint8))
