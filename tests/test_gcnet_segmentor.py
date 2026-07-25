import numpy as np
import pytest

from visual_deploy.segmentation.gcnet_segmentor import GCNetSegmentor, SegmentResult


class FakeGCNetSession:
    input_names = ("bgr", "depth_mm")
    output_names = ("prob",)

    def __init__(self, prob=None):
        self.prob = np.full((1, 1, 256, 256), 0.75, dtype=np.float32) if prob is None else prob
        self.inputs = None

    def run(self, inputs):
        self.inputs = inputs
        return {"prob": self.prob}


def _segmentor(session=None, threshold=0.5):
    return GCNetSegmentor("unused.onnx", threshold=threshold, session=session or FakeGCNetSession())


def test_gcnet_segmentor_returns_thresholded_numpy_result():
    session = FakeGCNetSession()
    out = _segmentor(session).infer(np.zeros((256, 256, 3), dtype=np.uint8), np.full((256, 256), 500.0))
    assert session.inputs["bgr"].shape == (1, 3, 256, 256)
    assert session.inputs["depth_mm"].shape == (1, 1, 256, 256)
    assert out.prob_256.shape == (256, 256)
    assert out.mask_256.dtype == bool
    assert out.mask_256.all()


@pytest.mark.parametrize(
    ("color_shape", "depth_shape", "message"),
    [
        ((128, 256, 3), (256, 256), "color_bgr_256"),
        ((256, 256, 1), (256, 256), "color_bgr_256"),
        ((256, 256, 3), (256, 128), "depth_mm_256"),
    ],
)
def test_gcnet_segmentor_rejects_wrong_input_shapes(color_shape, depth_shape, message):
    with pytest.raises(ValueError, match=message):
        _segmentor().infer(np.zeros(color_shape, dtype=np.uint8), np.zeros(depth_shape, dtype=np.float32))


def test_gcnet_segmentor_uses_configured_threshold_and_reports_component_ratio():
    prob = np.zeros((1, 1, 256, 256), dtype=np.float32)
    prob[:, :, 0:2, 0:2] = 0.8
    prob[:, :, 10:12, 10:13] = 0.8
    out = _segmentor(FakeGCNetSession(prob), threshold=0.7).infer(
        np.zeros((256, 256, 3), dtype=np.uint8), np.full((256, 256), 500.0)
    )
    assert out.mask_area == 10
    assert out.largest_component_ratio == pytest.approx(0.6)


def test_gcnet_segmentor_rejects_bad_output_shape_and_non_finite_values():
    color = np.zeros((256, 256, 3), dtype=np.uint8)
    depth = np.full((256, 256), 500.0)
    with pytest.raises(ValueError, match="prob output"):
        _segmentor(FakeGCNetSession(np.zeros((1, 256, 256, 1), dtype=np.float32))).infer(color, depth)
    with pytest.raises(ValueError, match="prob output"):
        _segmentor(FakeGCNetSession(np.full((1, 1, 256, 256), np.nan, dtype=np.float32))).infer(color, depth)


def test_segment_result_equality_is_identity_based_with_array_fields():
    result = SegmentResult(np.zeros((256, 256)), np.zeros((256, 256), dtype=bool), 0, 0.0)
    same_values = SegmentResult(np.zeros((256, 256)), np.zeros((256, 256), dtype=bool), 0, 0.0)
    assert result == result
    assert result != same_values


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_gcnet_segmentor_rejects_bad_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        _segmentor(threshold=threshold)
