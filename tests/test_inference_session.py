import ctypes

import numpy as np
import pytest

from visual_deploy.inference.cuda_runtime import CudaRuntime, CudaRuntimeError
from visual_deploy.inference.session import TensorRTEngineSession, _require_trt_success


class _FakeFunction:
    def __init__(self, callback):
        self.callback = callback
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self.callback(*args)


def _pointer_value(value):
    return value.value if hasattr(value, "value") else int(value)


class _FakeCudartLibrary:
    def __init__(self, *, device_count=1, set_device_result=0):
        self.device_count = device_count
        self.set_device_result = set_device_result
        self.selected_device = None
        self.next_pointer = 1000
        self.allocations = {}
        self.freed = []
        self.copies = []
        self.stream_flags = []
        self.synchronized = []
        self.destroyed = []
        self.cudaGetDeviceCount = _FakeFunction(self._get_device_count)
        self.cudaSetDevice = _FakeFunction(self._set_device)
        self.cudaMalloc = _FakeFunction(self._malloc)
        self.cudaFree = _FakeFunction(self._free)
        self.cudaMemcpyAsync = _FakeFunction(self._memcpy_async)
        self.cudaStreamCreateWithFlags = _FakeFunction(self._stream_create)
        self.cudaStreamSynchronize = _FakeFunction(self._stream_synchronize)
        self.cudaStreamDestroy = _FakeFunction(self._stream_destroy)
        self.cudaGetErrorString = _FakeFunction(lambda code: b"fake CUDA failure")

    def _get_device_count(self, result):
        result._obj.value = self.device_count
        return 0

    def _set_device(self, device_id):
        self.selected_device = _pointer_value(device_id)
        return self.set_device_result

    def _malloc(self, result, nbytes):
        pointer = self.next_pointer
        self.next_pointer += 64
        result._obj.value = pointer
        self.allocations[pointer] = _pointer_value(nbytes)
        return 0

    def _free(self, pointer):
        self.freed.append(_pointer_value(pointer))
        return 0

    def _memcpy_async(self, destination, source, nbytes, kind, stream):
        self.copies.append(
            (
                _pointer_value(destination),
                _pointer_value(source),
                _pointer_value(nbytes),
                _pointer_value(kind),
                _pointer_value(stream),
            )
        )
        return 0

    def _stream_create(self, result, flags):
        result._obj.value = 9000
        self.stream_flags.append(_pointer_value(flags))
        return 0

    def _stream_synchronize(self, stream):
        self.synchronized.append(_pointer_value(stream))
        return 0

    def _stream_destroy(self, stream):
        self.destroyed.append(_pointer_value(stream))
        return 0


class _FakeDeviceBuffer:
    def __init__(self, runtime, pointer, nbytes):
        self.runtime = runtime
        self.pointer = pointer
        self.nbytes = nbytes
        self.data = None

    def close(self):
        if self.pointer == 0:
            return
        self.runtime.freed.append(self.pointer)
        self.pointer = 0


class _FakeStream:
    def __init__(self, runtime):
        self.runtime = runtime
        self.handle = 1234
        self.synchronize_calls = 0

    def synchronize(self):
        self.synchronize_calls += 1

    def close(self):
        if self.handle:
            self.runtime.destroyed_streams += 1
            self.handle = 0


class _FakeCudaRuntime:
    def __init__(self):
        self.registry = {}
        self.allocations = 0
        self.freed = []
        self.stream_creations = 0
        self.destroyed_streams = 0
        self.host_to_device_copies = 0
        self.device_to_host_copies = 0
        self._next_pointer = 100
        self.created_stream = None
        self.fail_on_allocation = None

    def allocate(self, nbytes):
        if self.fail_on_allocation == self.allocations + 1:
            raise CudaRuntimeError("injected allocation failure")
        pointer = self._next_pointer
        self._next_pointer += 1
        buffer = _FakeDeviceBuffer(self, pointer, nbytes)
        self.registry[pointer] = buffer
        self.allocations += 1
        return buffer

    def create_stream(self):
        self.stream_creations += 1
        self.created_stream = _FakeStream(self)
        return self.created_stream

    def copy_host_to_device_async(self, destination, source, stream):
        self.host_to_device_copies += 1
        destination.data = np.array(source, copy=True)

    def copy_device_to_host_async(self, destination, source, stream):
        self.device_to_host_copies += 1
        destination[...] = source.data


class _FakeTrt:
    @staticmethod
    def nptype(dtype):
        return dtype


class _TensorEngine:
    def get_tensor_dtype(self, name):
        return np.float32


class _TensorContext:
    def __init__(self, cuda):
        self.cuda = cuda
        self.shape = None
        self.addresses = {}
        self.set_shape_calls = 0
        self.set_address_calls = 0
        self.execute_calls = 0
        self.execute_result = True

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
        if not self.execute_result:
            return False
        source = self.cuda.registry[self.addresses["input"]]
        output = self.cuda.registry[self.addresses["output"]]
        output.data = source.data * 2.0
        return True


class _BindingEngine:
    num_bindings = 2

    def get_binding_index(self, name):
        return {"input": 0, "output": 1}[name]

    def get_binding_shape(self, index):
        return (-1, 2)

    def get_binding_dtype(self, index):
        return np.float32


class _BindingContext:
    def __init__(self, cuda):
        self.cuda = cuda
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
        source = self.cuda.registry[addresses[0]]
        output = self.cuda.registry[addresses[1]]
        output.data = source.data + 3.0
        return True


def _fake_session(*, tensor_api, reuse_buffers=True):
    cuda = _FakeCudaRuntime()
    session = TensorRTEngineSession.__new__(TensorRTEngineSession)
    session.trt = _FakeTrt()
    session.cuda = cuda
    session.engine = _TensorEngine() if tensor_api else _BindingEngine()
    session.context = _TensorContext(cuda) if tensor_api else _BindingContext(cuda)
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


def test_cuda_runtime_owns_allocations_copies_and_nonblocking_stream():
    library = _FakeCudartLibrary(device_count=2)
    runtime = CudaRuntime(1, library=library)
    source = np.array([1.0, 2.0], dtype=np.float32)
    destination = np.empty_like(source)
    buffer = runtime.allocate(source.nbytes)
    stream = runtime.create_stream()

    runtime.copy_host_to_device_async(buffer, source, stream)
    runtime.copy_device_to_host_async(destination, buffer, stream)
    stream.synchronize()
    buffer.close()
    buffer.close()
    stream.close()
    stream.close()

    assert library.selected_device == 1
    assert library.stream_flags == [CudaRuntime.STREAM_NON_BLOCKING]
    assert [copy[3] for copy in library.copies] == [
        CudaRuntime.MEMCPY_HOST_TO_DEVICE,
        CudaRuntime.MEMCPY_DEVICE_TO_HOST,
    ]
    assert library.synchronized == [9000]
    assert library.freed == [1000]
    assert library.destroyed == [9000]


def test_cuda_runtime_reports_api_name_code_and_error_text():
    library = _FakeCudartLibrary(set_device_result=10)

    with pytest.raises(
        CudaRuntimeError,
        match=r"cudaSetDevice\(0\) failed with CUDA error 10: fake CUDA failure",
    ):
        CudaRuntime(0, library=library)


def test_tensorrt_success_guard_accepts_only_explicit_true():
    _require_trt_success(True, "bind")

    for result in (False, None, 0):
        with pytest.raises(RuntimeError, match="TensorRT bind failed"):
            _require_trt_success(result, "bind")


def test_tensorrt_session_close_releases_buffers_stream_and_native_handles_once():
    session = _fake_session(tensor_api=True)
    session.run({"input": np.array([[1.0, 2.0]], dtype=np.float32)})

    session.close()
    session.close()

    assert session.context is None
    assert session.engine is None
    assert session.runtime is None
    assert len(session.cuda.freed) == 2
    assert session.cuda.destroyed_streams == 1


def test_tensor_api_reuses_buffers_and_rebuilds_only_after_shape_change():
    session = _fake_session(tensor_api=True)

    first = session.run({"input": np.array([[1.0, 2.0]], dtype=np.float32)})["output"]
    second = session.run({"input": np.array([[3.0, 4.0]], dtype=np.float32)})["output"]

    assert np.array_equal(first, [[2.0, 4.0]])
    assert np.array_equal(second, [[6.0, 8.0]])
    assert np.array_equal(first, [[2.0, 4.0]])
    assert not np.shares_memory(first, second)
    assert session.cuda.allocations == 2
    assert session.context.set_shape_calls == 1
    assert session.context.set_address_calls == 2
    assert session.context.execute_calls == 2
    assert session.buffer_rebuild_count == 1
    assert session.cuda.stream_creations == 1

    session.run({"input": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)})

    assert session.cuda.allocations == 4
    assert len(session.cuda.freed) == 2
    assert session.context.set_shape_calls == 2
    assert session.context.set_address_calls == 4
    assert session.buffer_rebuild_count == 2


def test_binding_api_reuses_buffers_and_addresses_for_stable_shape():
    session = _fake_session(tensor_api=False)

    first = session.run({"input": np.array([[1.0, 2.0]], dtype=np.float32)})["output"]
    addresses = session._binding_addresses
    second = session.run({"input": np.array([[4.0, 5.0]], dtype=np.float32)})["output"]

    assert np.array_equal(first, [[4.0, 5.0]])
    assert np.array_equal(second, [[7.0, 8.0]])
    assert session._binding_addresses is addresses
    assert session.cuda.allocations == 2
    assert session.context.set_shape_calls == 1
    assert session.context.execute_calls == 2
    assert session.buffer_rebuild_count == 1


def test_buffer_reuse_rejects_input_dtype_mismatch_before_execution():
    session = _fake_session(tensor_api=True)

    with pytest.raises(TypeError, match="expects float32, got float64"):
        session.run({"input": np.array([[1.0, 2.0]], dtype=np.float64)})

    assert session.context.execute_calls == 0
    assert session.buffer_rebuild_count == 0
    assert session.cuda.allocations == 0


def test_partial_buffer_build_failure_releases_completed_allocations():
    session = _fake_session(tensor_api=True)
    session.cuda.fail_on_allocation = 2

    with pytest.raises(CudaRuntimeError, match="injected allocation failure"):
        session.run({"input": np.array([[1.0, 2.0]], dtype=np.float32)})

    assert session.cuda.allocations == 1
    assert len(session.cuda.freed) == 1
    assert session._buffers == {}
    assert session.buffer_rebuild_count == 0


def test_disabled_buffer_reuse_allocates_and_frees_each_call_but_reuses_stream():
    session = _fake_session(tensor_api=True, reuse_buffers=False)

    session.run({"input": np.array([[1.0, 2.0]], dtype=np.float32)})
    session.run({"input": np.array([[3.0, 4.0]], dtype=np.float32)})

    assert session.cuda.allocations == 4
    assert len(session.cuda.freed) == 4
    assert session.cuda.stream_creations == 1
    assert session.context.set_shape_calls == 2
    assert session.context.set_address_calls == 4
    assert session.buffer_rebuild_count == 0


def test_execution_failure_synchronizes_stream_before_nonreused_buffers_are_freed():
    session = _fake_session(tensor_api=True, reuse_buffers=False)
    session.context.execute_result = False

    with pytest.raises(RuntimeError, match="execute_async_v3 failed"):
        session.run({"input": np.array([[1.0, 2.0]], dtype=np.float32)})

    assert session.cuda.created_stream.synchronize_calls == 1
    assert len(session.cuda.freed) == 2
