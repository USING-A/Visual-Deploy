import numpy as np
import pytest
import torch

from visual_deploy.segmentation.gcnet_segmentor import GCNetSegmentor, SegmentResult


pytestmark = pytest.mark.filterwarnings("ignore:`torch.jit.:DeprecationWarning")


class TinyScriptableSegmentor(torch.nn.Module):
    def forward(self, bgr, depth_mm):
        prob = torch.ones((1, 1, 256, 256), dtype=torch.float32) * 0.75
        mask = (prob >= 0.5).to(torch.uint8)
        return prob, mask


class Tiny255MaskSegmentor(torch.nn.Module):
    def forward(self, bgr, depth_mm):
        prob = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        mask = torch.zeros((1, 1, 256, 256), dtype=torch.uint8)
        mask[:, :, 32:96, 32:96] = 255
        return prob, mask


class TinyNHWCSegmentor(torch.nn.Module):
    def forward(self, bgr, depth_mm):
        prob = torch.ones((1, 256, 256, 1), dtype=torch.float32)
        mask = torch.ones((1, 256, 256, 1), dtype=torch.uint8)
        return prob, mask


class TinyNonFiniteProbSegmentor(torch.nn.Module):
    def forward(self, bgr, depth_mm):
        prob = torch.full((1, 1, 256, 256), float("nan"), dtype=torch.float32)
        mask = torch.ones((1, 1, 256, 256), dtype=torch.uint8)
        return prob, mask


class TinyTwoComponentSegmentor(torch.nn.Module):
    def forward(self, bgr, depth_mm):
        prob = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        mask = torch.zeros((1, 1, 256, 256), dtype=torch.uint8)
        mask[:, :, 0:2, 0:2] = 1
        mask[:, :, 10:12, 10:13] = 1
        return prob, mask


def _save_traced(module: torch.nn.Module, path) -> None:
    scripted = torch.jit.trace(
        module,
        (torch.zeros(1, 3, 256, 256), torch.zeros(1, 1, 256, 256)),
    )
    scripted.save(str(path))


def test_gcnet_segmentor_loads_torchscript_and_returns_numpy(tmp_path):
    path = tmp_path / "tiny.pt"
    _save_traced(TinyScriptableSegmentor(), path)
    seg = GCNetSegmentor(path, device="cpu")
    color = np.zeros((256, 256, 3), dtype=np.uint8)
    depth = np.ones((256, 256), dtype=np.float32) * 500.0
    out = seg.infer(color, depth)
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
def test_gcnet_segmentor_rejects_wrong_input_shapes(tmp_path, color_shape, depth_shape, message):
    path = tmp_path / "tiny.pt"
    _save_traced(TinyScriptableSegmentor(), path)
    seg = GCNetSegmentor(path, device="cpu")
    color = np.zeros(color_shape, dtype=np.uint8)
    depth = np.zeros(depth_shape, dtype=np.float32)

    with pytest.raises(ValueError, match=message):
        seg.infer(color, depth)


def test_gcnet_segmentor_converts_255_mask_to_bool_and_reports_area(tmp_path):
    path = tmp_path / "tiny255.pt"
    _save_traced(Tiny255MaskSegmentor(), path)
    seg = GCNetSegmentor(path, device="cpu")
    color = np.zeros((256, 256, 3), dtype=np.uint8)
    depth = np.ones((256, 256), dtype=np.float32) * 500.0

    out = seg.infer(color, depth)

    assert out.mask_256.dtype == bool
    assert out.mask_256.sum() == 64 * 64
    assert out.mask_area == 64 * 64
    assert out.largest_component_ratio == pytest.approx(1.0)


def test_gcnet_segmentor_rejects_nhwc_model_output(tmp_path):
    path = tmp_path / "tiny_nhwc.pt"
    _save_traced(TinyNHWCSegmentor(), path)
    seg = GCNetSegmentor(path, device="cpu")
    color = np.zeros((256, 256, 3), dtype=np.uint8)
    depth = np.ones((256, 256), dtype=np.float32) * 500.0

    with pytest.raises(ValueError, match="prob output"):
        seg.infer(color, depth)


def test_gcnet_segmentor_rejects_non_finite_prob_output(tmp_path):
    path = tmp_path / "tiny_nan.pt"
    _save_traced(TinyNonFiniteProbSegmentor(), path)
    seg = GCNetSegmentor(path, device="cpu")
    color = np.zeros((256, 256, 3), dtype=np.uint8)
    depth = np.ones((256, 256), dtype=np.float32) * 500.0

    with pytest.raises(ValueError, match="prob output"):
        seg.infer(color, depth)


def test_segment_result_equality_is_identity_based_with_array_fields():
    result = SegmentResult(
        prob_256=np.zeros((256, 256), dtype=np.float32),
        mask_256=np.zeros((256, 256), dtype=bool),
        mask_area=0,
        largest_component_ratio=0.0,
    )
    same_values = SegmentResult(
        prob_256=np.zeros((256, 256), dtype=np.float32),
        mask_256=np.zeros((256, 256), dtype=bool),
        mask_area=0,
        largest_component_ratio=0.0,
    )

    assert result == result
    assert result != same_values


def test_gcnet_segmentor_reports_largest_component_ratio(tmp_path):
    path = tmp_path / "tiny_components.pt"
    _save_traced(TinyTwoComponentSegmentor(), path)
    seg = GCNetSegmentor(path, device="cpu")
    color = np.zeros((256, 256, 3), dtype=np.uint8)
    depth = np.ones((256, 256), dtype=np.float32) * 500.0

    out = seg.infer(color, depth)

    assert out.mask_area == 10
    assert out.largest_component_ratio == pytest.approx(0.6)


def test_gcnet_segmentor_warns_and_uses_cpu_when_cuda_unavailable(tmp_path, monkeypatch):
    path = tmp_path / "tiny.pt"
    _save_traced(TinyScriptableSegmentor(), path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.warns(RuntimeWarning, match="CUDA requested"):
        seg = GCNetSegmentor(path, device="cuda")

    assert seg.device.type == "cpu"


@pytest.mark.parametrize(
    ("color", "depth", "message"),
    [
        (np.full((256, 256, 3), np.nan, dtype=np.float32), np.zeros((256, 256), dtype=np.float32), "color_bgr_256"),
        (np.zeros((256, 256, 3), dtype=np.uint8), np.full((256, 256), np.inf, dtype=np.float32), "depth_mm_256"),
    ],
)
def test_gcnet_segmentor_rejects_non_finite_inputs(tmp_path, color, depth, message):
    path = tmp_path / "tiny.pt"
    _save_traced(TinyScriptableSegmentor(), path)
    seg = GCNetSegmentor(path, device="cpu")

    with pytest.raises(ValueError, match=message):
        seg.infer(color, depth)
