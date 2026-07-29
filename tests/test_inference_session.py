import pytest

from visual_deploy.inference.session import TensorRTEngineSession, _require_trt_success


def test_tensorrt_success_guard_accepts_only_explicit_true():
    _require_trt_success(True, "bind")

    for result in (False, None, 0):
        with pytest.raises(RuntimeError, match="TensorRT bind failed"):
            _require_trt_success(result, "bind")


def test_tensorrt_session_close_releases_native_handles_and_cuda_cache():
    class FakeCuda:
        def __init__(self):
            self.empty_cache_calls = 0

        def empty_cache(self):
            self.empty_cache_calls += 1

    session = TensorRTEngineSession.__new__(TensorRTEngineSession)
    session._closed = False
    session.context = object()
    session.engine = object()
    session.runtime = object()
    session.torch = type("FakeTorch", (), {"cuda": FakeCuda()})()

    session.close()
    session.close()

    assert session.context is None
    assert session.engine is None
    assert session.runtime is None
    assert session.torch.cuda.empty_cache_calls == 1
