import warnings

import pytest
import torch

from scripts.export_gcnet_l03_inference import (
    RGBDGCNetInferenceWrapper,
    assert_wrapper_parity,
    normalize_bgr_depth,
)


class _FakeBackbone(torch.nn.Module):
    def forward(self, x):
        return x


class _FakeDecodeHead(torch.nn.Module):
    align_corners = False

    def forward(self, x):
        bg = -x[:, 0:1] - 0.2 * x[:, 3:4]
        fg = x[:, 0:1] + 0.2 * x[:, 3:4]
        return torch.cat([bg, fg], dim=1)


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _FakeBackbone()
        self.decode_head = _FakeDecodeHead()


def test_normalize_bgr_depth_converts_bgr_to_rgb_and_depth_mm_to_m():
    bgr = torch.zeros(1, 3, 2, 2)
    bgr[:, 0] = 10.0
    bgr[:, 1] = 20.0
    bgr[:, 2] = 30.0
    depth_mm = torch.full((1, 1, 2, 2), 500.0)
    out = normalize_bgr_depth(bgr, depth_mm)
    assert out.shape == (1, 4, 2, 2)
    expected_r = (30.0 - 123.675) / 58.395
    expected_g = (20.0 - 116.28) / 57.12
    expected_b = (10.0 - 103.53) / 57.375
    assert out[0, 0, 0, 0].item() == torch.tensor(expected_r).item()
    assert out[0, 1, 0, 0].item() == torch.tensor(expected_g).item()
    assert out[0, 2, 0, 0].item() == torch.tensor(expected_b).item()
    assert out[0, 3, 0, 0].item() == 0.5


def test_normalize_bgr_depth_accepts_unbatched_inputs():
    bgr = torch.zeros(3, 2, 3)
    depth_mm = torch.full((2, 3), 250.0)

    out = normalize_bgr_depth(bgr, depth_mm)

    assert out.shape == (1, 4, 2, 3)
    assert out[0, 3, 0, 0].item() == 0.25


@pytest.mark.parametrize(
    ("bgr_shape", "depth_shape", "message"),
    [
        ((1, 2, 4, 4), (1, 1, 4, 4), "3 channels"),
        ((1, 3, 4, 4), (2, 1, 4, 4), "batch"),
        ((1, 3, 4, 4), (1, 1, 3, 4), "spatial"),
        ((1, 3, 4, 4), (1, 2, 4, 4), "1 channel"),
        ((4, 4), (4, 4), "BGR"),
    ],
)
def test_normalize_bgr_depth_rejects_bad_shapes(bgr_shape, depth_shape, message):
    bgr = torch.zeros(bgr_shape)
    depth_mm = torch.zeros(depth_shape)

    with pytest.raises(ValueError, match=message):
        normalize_bgr_depth(bgr, depth_mm)


def test_inference_wrapper_registers_preprocessing_constants_as_buffers():
    wrapper = RGBDGCNetInferenceWrapper(_FakeModel())

    buffers = dict(wrapper.named_buffers())

    assert set(buffers) >= {"rgb_mean", "rgb_std"}
    assert buffers["rgb_mean"].shape == (1, 3, 1, 1)
    assert buffers["rgb_std"].shape == (1, 3, 1, 1)


def test_inference_wrapper_outputs_foreground_probability_and_uint8_mask():
    wrapper = RGBDGCNetInferenceWrapper(_FakeModel(), threshold=0.5)
    bgr = torch.zeros(2, 3, 2, 2)
    bgr[:, 2] = 255.0
    depth_mm = torch.full((2, 1, 2, 2), 500.0)

    foreground, mask = wrapper(bgr, depth_mm)

    assert foreground.shape == (2, 1, 2, 2)
    assert foreground.dtype == torch.float32
    assert mask.shape == (2, 1, 2, 2)
    assert mask.dtype == torch.uint8
    assert set(torch.unique(mask).tolist()) <= {0, 255}


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), float("inf")])
def test_inference_wrapper_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        RGBDGCNetInferenceWrapper(_FakeModel(), threshold=threshold)


@pytest.mark.parametrize(
    ("bgr", "depth_mm", "message"),
    [
        (torch.zeros(3, 2, 2), torch.zeros(1, 1, 2, 2), "BGR"),
        (torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2), "depth"),
        (torch.zeros(1, 2, 2, 2), torch.zeros(1, 1, 2, 2), "3 channels"),
        (torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2, 2), "1 channel"),
    ],
)
def test_inference_wrapper_requires_batched_4d_inputs(bgr, depth_mm, message):
    wrapper = RGBDGCNetInferenceWrapper(_FakeModel())

    with pytest.raises(ValueError, match=message):
        wrapper(bgr, depth_mm)


def test_assert_wrapper_parity_compares_eager_and_traced_outputs():
    wrapper = RGBDGCNetInferenceWrapper(_FakeModel(), threshold=0.4)
    wrapper.eval()
    bgr = torch.linspace(0.0, 255.0, steps=2 * 3 * 4 * 4, dtype=torch.float32).reshape(2, 3, 4, 4)
    depth_mm = torch.linspace(250.0, 750.0, steps=2 * 1 * 4 * 4, dtype=torch.float32).reshape(2, 1, 4, 4)

    with torch.no_grad():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="`torch.jit.trace", category=DeprecationWarning)
            warnings.filterwarnings("ignore", message="`torch.jit.trace_method", category=DeprecationWarning)
            traced = torch.jit.trace(wrapper, (bgr, depth_mm), strict=False)

    assert_wrapper_parity(wrapper, traced, bgr, depth_mm)
