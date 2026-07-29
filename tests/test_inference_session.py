from contextlib import nullcontext

import numpy as np
import pytest

from visual_deploy.inference.session import TensorRTEngineSession, _require_trt_success


class _FakeCpuTensor:
    def __init__(self, array):
        self._array = np.array(array, copy=True)

    def numpy(self):
        return self._array


class _FakeTensor:
    def __init__(self, shape, dtype, registry, pointer):
        self.array = np.zeros(shape, dtype=dtype)
        self._registry = registry
        self._pointer = pointer
        registry[pointer] = self

    @property
    def shape(self):
        return self.array.shape

    def data_ptr(self):
        return self._pointer

    def copy_(self, other):
        self.array[...] = other.array
        return self

    def contiguous(self):
        return self

    def detach(self):
        return self

    def cpu(self):
        return _FakeCpuTensor(self.array)


class _FakeHostTensor:
    def __init__(self, array):
        self.array = np.asarray(array)


class _FakeStream:
    cuda_stream = 1234

    def __init__(self):
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1


class _FakeCuda:
    def __init__(self):
        self.stream_creations = 0
        self.empty_cache_calls = 0
        self.created_stream = None

    def Stream(self, device=None):
        self.stream_creations += 1
        self.created_stream = _FakeStream()
        return self.created_stream

    def stream(self, stream):
        return nullcontext()

    def current_stream(self, device=None):
        if self.created_stream is None:
            self.created_stream = _FakeStream()
        return self.created_stream

    def empty_cache(self):
        self.empty_cache_calls += 1


class _FakeTorch:
    float16 = np.float16
    float32 = np.float32
    int8 = np.int8
    int32 = np.int32
    int64 = np.int64
    uint8 = np.uint8
    bool = np.bool_

    def __init__(self):
        self.cuda = _FakeCuda()
        self.registry = {}
        self.empty_calls = 0
        self.as_tensor_calls = 0
        self._next_pointer = 100

    def empty(self, shape, dtype, device=None):
        self.empty_calls += 1
        pointer = self._next_pointer
        self._next_pointer += 1
        return _FakeTensor(shape, dtype, self.registry, pointer)

    def from_numpy(self, array):
        return _FakeHostTensor(array)

    def as_tensor(self, array, device=None):
        self.as_tensor_calls += 1
        tensor = self.empty(np.asarray(array).shape, np.asarray(array).dtype, device=device)
        tensor.array[...] = array
        return tensor


class _FakeTrt:
    @staticmethod
    def nptype(dtype):
        return dtype


class _TensorEngine:
    def get_tensor_dtype(self, name):
        return np.float32


class _TensorContext:
    def __init__(self, torch):
        self.torch = torch
        self.shape = None
        self.addresses = {}
        self.set_shape_calls = 0
        self.set_address_calls = 0
        self.execute_calls = 0

    def set_input_shape(self, name, shape):
        self.shape = tuple(shape)
        self.set_shape_calls += 1
        return True

    def get_tensor_shape(self, name):
        return self.shape

    def set_tensor_address(self, name, pointer):
        self.addresses[name] = pointer
        self.set_address_calls += 1
        return True

    def execute_async_v3(self, stream_handle):
        self.execute_calls += 1
        source = self.torch.registry[self.addresses["input"]]
        output = self.torch.registry[self.addresses["output"]]
        output.array[...] = source.array * 2.0
        return True


class _BindingEngine:
    num_bindings = 2

    def get_binding_index(self, name):
        return {"input": 0, "output": 1}[name]

    def get_binding_shape(self, index):
        return (-1, 2) if index == 0 else (-1, 2)

    def get_binding_dtype(self, index):
        return np.float32


class _BindingContext:
    def __init__(self, torch):
        self.torch = torch
        self.shape = None
        self.set_shape_calls = 0
        self.execute_calls = 0

    def set_binding_shape(self, index, shape):
        self.shape = tuple(shape)
        self.set_shape_calls += 1
        return True

    def get_binding_shape(self, index):
        return self.shape

    def execute_async_v2(self, addresses, stream_handle):
        self.execute_calls += 1
        source = self.torch.registry[addresses[0]]
        output = self.torch.registry[addresses[1]]
        output.array[...] = source.array + 3.0
        return True


def _fake_session(*, tensor_api, reuse_buffers=True):
    torch = _FakeTorch()
    session = TensorRTEngineSession.__new__(TensorRTEngineSession)
    session.trt = _FakeTrt()
    session.torch = torch
    session.device = "cuda:0"
    session.engine = _TensorEngine() if tensor_api else _BindingEngine()
    session.context = _TensorContext(torch) if tensor_api else _BindingContext(torch)
    session.runtime = object()
    session._tensor_api = tensor_api
    session._input_names = ("input",)
    session._output_names = ("output",)
    session.reuse_buffers = reuse_buffers
    session._buffer_signature = None
    session._buffers = {}
    session._binding_addresses = None
    session._execution_stream = None
    session._buffer_rebuild_count = 0
    session._closed = False
    return session


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


def test_tensor_api_reuses_buffers_and_rebuilds_only_after_shape_change():
    session = _fake_session(tensor_api=True)

    first = session.run({"input": np.array([[1.0, 2.0]], dtype=np.float32)})["output"]
    second = session.run({"input": np.array([[3.0, 4.0]], dtype=np.float32)})["output"]

    assert np.array_equal(first, [[2.0, 4.0]])
    assert np.array_equal(second, [[6.0, 8.0]])
    assert np.array_equal(first, [[2.0, 4.0]])
    assert not np.shares_memory(first, second)
    assert session.torch.empty_calls == 2
    assert session.context.set_shape_calls == 1
    assert session.context.set_address_calls == 2
    assert session.context.execute_calls == 2
    assert session.buffer_rebuild_count == 1
    assert session.torch.cuda.stream_creations == 1

    session.run({"input": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)})

    assert session.torch.empty_calls == 4
    assert session.context.set_shape_calls == 2
    assert session.context.set_address_calls == 4
    assert session.buffer_rebuild_count == 2

    session.close()

    assert session._buffers == {}
    assert session._binding_addresses is None
    assert session._execution_stream is None


def test_binding_api_reuses_buffers_and_addresses_for_stable_shape():
    session = _fake_session(tensor_api=False)

    first = session.run({"input": np.array([[1.0, 2.0]], dtype=np.float32)})["output"]
    addresses = session._binding_addresses
    second = session.run({"input": np.array([[4.0, 5.0]], dtype=np.float32)})["output"]

    assert np.array_equal(first, [[4.0, 5.0]])
    assert np.array_equal(second, [[7.0, 8.0]])
    assert session._binding_addresses is addresses
    assert session.torch.empty_calls == 2
    assert session.context.set_shape_calls == 1
    assert session.context.execute_calls == 2
    assert session.buffer_rebuild_count == 1


def test_buffer_reuse_rejects_input_dtype_mismatch_before_execution():
    session = _fake_session(tensor_api=True)

    with pytest.raises(TypeError, match="expects float32, got float64"):
        session.run({"input": np.array([[1.0, 2.0]], dtype=np.float64)})

    assert session.context.execute_calls == 0
    assert session.buffer_rebuild_count == 0


def test_disabled_buffer_reuse_keeps_per_call_allocation_path():
    session = _fake_session(tensor_api=True, reuse_buffers=False)

    session.run({"input": np.array([[1.0, 2.0]], dtype=np.float32)})
    session.run({"input": np.array([[3.0, 4.0]], dtype=np.float32)})

    assert session.torch.as_tensor_calls == 2
    assert session.torch.empty_calls == 4
    assert session.context.set_shape_calls == 2
    assert session.context.set_address_calls == 4
    assert session.buffer_rebuild_count == 0
