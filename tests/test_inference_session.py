import pytest

from visual_deploy.inference.session import _require_trt_success


def test_tensorrt_success_guard_accepts_only_explicit_true():
    _require_trt_success(True, "bind")

    for result in (False, None, 0):
        with pytest.raises(RuntimeError, match="TensorRT bind failed"):
            _require_trt_success(result, "bind")
