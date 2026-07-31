from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .cuda_runtime import CudaDeviceBuffer, CudaRuntime, CudaStream


class InferenceSession(Protocol):
    @property
    def input_names(self) -> tuple[str, ...]: ...

    @property
    def output_names(self) -> tuple[str, ...]: ...

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...

    def close(self) -> None: ...


class OnnxRuntimeSession:
    def __init__(self, model_path: str | PathLike[str], device: str = "cpu") -> None:
        path = _validate_model_path(model_path, ".onnx")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required for ONNX inference") from exc

        available = set(ort.get_available_providers())
        requested = str(device).lower()
        if requested.startswith("cuda") and "CUDAExecutionProvider" in available:
            providers: list[Any] = [
                ("CUDAExecutionProvider", {"device_id": _cuda_device_id(requested)}),
                "CPUExecutionProvider",
            ]
        else:
            providers = ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(path), providers=providers)
        self._closed = False
        self._input_names = tuple(item.name for item in self.session.get_inputs())
        self._output_names = tuple(item.name for item in self.session.get_outputs())

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._closed:
            raise RuntimeError("ONNX Runtime session is closed")
        _validate_input_names(inputs, self.input_names)
        arrays = {name: np.ascontiguousarray(inputs[name]) for name in self.input_names}
        outputs = self.session.run(list(self.output_names), arrays)
        return dict(zip(self.output_names, outputs))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        session = self.session
        self.session = None
        del session


@dataclass
class _TensorBuffer:
    device: CudaDeviceBuffer
    shape: tuple[int, ...]
    dtype: np.dtype[Any]

    @property
    def pointer(self) -> int:
        return self.device.pointer

    def empty_host_array(self) -> np.ndarray:
        return np.empty(self.shape, dtype=self.dtype)


class TensorRTEngineSession:
    """Load a TensorRT engine and execute it with JetPack's CUDA Runtime.

    TensorRT 8's binding API and TensorRT 10's named-tensor API are both
    supported. No PyTorch, PyCUDA, CuPy, or extra CUDA Python package is used.
    """

    def __init__(
        self,
        model_path: str | PathLike[str],
        device: str = "cuda:0",
        *,
        reuse_buffers: bool = True,
        cuda_runtime: CudaRuntime | None = None,
    ) -> None:
        path = _validate_model_path(model_path, ".engine")
        if not isinstance(reuse_buffers, bool):
            raise ValueError("reuse_buffers must be a boolean")
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError("TensorRT engine inference requires the JetPack tensorrt Python package") from exc

        self.trt = trt
        self.cuda = cuda_runtime or CudaRuntime(_cuda_device_id(str(device).lower()))
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {path}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"failed to create TensorRT execution context: {path}")
        self._tensor_api = hasattr(self.engine, "num_io_tensors")
        self._input_names, self._output_names = self._discover_io()
        self.reuse_buffers = reuse_buffers
        self._buffer_signature: tuple[tuple[str, tuple[int, ...], str], ...] | None = None
        self._buffers: dict[str, _TensorBuffer] = {}
        self._binding_addresses: list[int] | None = None
        self._execution_stream: CudaStream | None = None
        self._buffer_rebuild_count = 0
        self._closed = False

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    @property
    def buffer_rebuild_count(self) -> int:
        return self._buffer_rebuild_count

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if self._closed:
            raise RuntimeError("TensorRT engine session is closed")
        _validate_input_names(inputs, self.input_names)
        arrays = _contiguous_arrays(inputs, self.input_names)
        signature = _input_signature(arrays, self.input_names)

        if self.reuse_buffers:
            if signature != self._buffer_signature:
                self._clear_buffer_cache()
                self._buffers, self._binding_addresses = self._allocate_buffers(arrays)
                self._buffer_signature = signature
                self._buffer_rebuild_count += 1
            return self._execute(arrays, self._buffers, self._binding_addresses)

        buffers, addresses = self._allocate_buffers(arrays)
        try:
            return self._execute(arrays, buffers, addresses)
        finally:
            _release_buffers(buffers)

    def _discover_io(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self._tensor_api:
            names = [self.engine.get_tensor_name(index) for index in range(self.engine.num_io_tensors)]
            inputs = [name for name in names if self.engine.get_tensor_mode(name) == self.trt.TensorIOMode.INPUT]
            outputs = [name for name in names if name not in inputs]
            return tuple(inputs), tuple(outputs)
        inputs, outputs = [], []
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            (inputs if self.engine.binding_is_input(index) else outputs).append(name)
        return tuple(inputs), tuple(outputs)

    def _allocate_buffers(
        self,
        arrays: dict[str, np.ndarray],
    ) -> tuple[dict[str, _TensorBuffer], list[int] | None]:
        if self._tensor_api:
            return self._allocate_tensor_api_buffers(arrays), None
        return self._allocate_binding_api_buffers(arrays)

    def _allocate_tensor_api_buffers(self, arrays: dict[str, np.ndarray]) -> dict[str, _TensorBuffer]:
        buffers: dict[str, _TensorBuffer] = {}
        try:
            for name in self.input_names:
                array = arrays[name]
                expected_dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
                _validate_input_dtype(name, array.dtype, expected_dtype)
                if not self.context.set_input_shape(name, tuple(array.shape)):
                    raise ValueError(f"TensorRT rejected input shape for {name}: {tuple(array.shape)}")
                buffers[name] = self._allocate_tensor(tuple(array.shape), expected_dtype)
            for name in self.output_names:
                shape = _resolved_shape(name, self.context.get_tensor_shape(name))
                dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
                buffers[name] = self._allocate_tensor(shape, dtype)
            for name, buffer in buffers.items():
                _require_trt_success(
                    self.context.set_tensor_address(name, buffer.pointer),
                    f"set_tensor_address({name})",
                )
            return buffers
        except Exception:
            _release_buffers(buffers)
            raise

    def _allocate_binding_api_buffers(
        self,
        arrays: dict[str, np.ndarray],
    ) -> tuple[dict[str, _TensorBuffer], list[int]]:
        buffers: dict[str, _TensorBuffer] = {}
        addresses = [0] * self.engine.num_bindings
        try:
            for name in self.input_names:
                index = self.engine.get_binding_index(name)
                array = arrays[name]
                expected_dtype = np.dtype(self.trt.nptype(self.engine.get_binding_dtype(index)))
                _validate_input_dtype(name, array.dtype, expected_dtype)
                if -1 in tuple(self.engine.get_binding_shape(index)):
                    _require_trt_success(
                        self.context.set_binding_shape(index, tuple(array.shape)),
                        f"set_binding_shape({name})",
                    )
                buffer = self._allocate_tensor(tuple(array.shape), expected_dtype)
                buffers[name] = buffer
                addresses[index] = buffer.pointer
            for name in self.output_names:
                index = self.engine.get_binding_index(name)
                shape = _resolved_shape(name, self.context.get_binding_shape(index))
                dtype = np.dtype(self.trt.nptype(self.engine.get_binding_dtype(index)))
                buffer = self._allocate_tensor(shape, dtype)
                buffers[name] = buffer
                addresses[index] = buffer.pointer
            return buffers, addresses
        except Exception:
            _release_buffers(buffers)
            raise

    def _allocate_tensor(self, shape: tuple[int, ...], dtype: np.dtype[Any]) -> _TensorBuffer:
        nbytes = _tensor_nbytes(shape, dtype)
        return _TensorBuffer(self.cuda.allocate(nbytes), shape, dtype)

    def _execute(
        self,
        arrays: dict[str, np.ndarray],
        buffers: dict[str, _TensorBuffer],
        addresses: list[int] | None,
    ) -> dict[str, np.ndarray]:
        stream = self._execution_stream_for_session()
        outputs = {name: buffers[name].empty_host_array() for name in self.output_names}
        try:
            for name in self.input_names:
                self.cuda.copy_host_to_device_async(buffers[name].device, arrays[name], stream)
            if self._tensor_api:
                if not self.context.execute_async_v3(stream_handle=stream.handle):
                    raise RuntimeError("TensorRT execute_async_v3 failed")
            else:
                if addresses is None:
                    raise RuntimeError("TensorRT binding addresses are unavailable")
                if not self.context.execute_async_v2(addresses, stream.handle):
                    raise RuntimeError("TensorRT execute_async_v2 failed")
            for name in self.output_names:
                self.cuda.copy_device_to_host_async(outputs[name], buffers[name].device, stream)
        finally:
            stream.synchronize()
        return outputs

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: Exception | None = None
        try:
            self._clear_buffer_cache()
        except Exception as exc:
            first_error = exc
        stream = self._execution_stream
        self._execution_stream = None
        if stream is not None:
            try:
                stream.close()
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        for name in ("context", "engine", "runtime"):
            resource = getattr(self, name, None)
            setattr(self, name, None)
            del resource
        if first_error is not None:
            raise first_error

    def _execution_stream_for_session(self) -> CudaStream:
        if self._execution_stream is None:
            self._execution_stream = self.cuda.create_stream()
        return self._execution_stream

    def _clear_buffer_cache(self) -> None:
        self._buffer_signature = None
        buffers = self._buffers
        self._buffers = {}
        self._binding_addresses = None
        _release_buffers(buffers)


def create_inference_session(
    model_path: str | PathLike[str],
    *,
    device: str = "cpu",
    backend: str = "auto",
    reuse_buffers: bool = True,
) -> InferenceSession:
    path = Path(model_path)
    selected = str(backend).lower()
    if selected == "auto":
        selected = {".onnx": "onnxruntime", ".engine": "tensorrt"}.get(path.suffix.lower(), "")
    if selected == "onnxruntime":
        return OnnxRuntimeSession(path, device=device)
    if selected == "tensorrt":
        return TensorRTEngineSession(path, device=device, reuse_buffers=reuse_buffers)
    raise ValueError("backend must be auto, onnxruntime, or tensorrt; model must end in .onnx or .engine")


def _validate_model_path(model_path: str | PathLike[str], expected_suffix: str) -> Path:
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"model file not found: {path}")
    if path.suffix.lower() != expected_suffix:
        raise ValueError(f"expected a {expected_suffix} model, got: {path}")
    return path


def _validate_input_names(inputs: dict[str, np.ndarray], expected: tuple[str, ...]) -> None:
    if set(inputs) != set(expected):
        raise ValueError(f"model inputs must be {list(expected)}, got {sorted(inputs)}")


def _contiguous_arrays(inputs: dict[str, np.ndarray], names: tuple[str, ...]) -> dict[str, np.ndarray]:
    return {name: np.ascontiguousarray(inputs[name]) for name in names}


def _input_signature(
    arrays: dict[str, np.ndarray],
    names: tuple[str, ...],
) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    return tuple((name, tuple(arrays[name].shape), arrays[name].dtype.str) for name in names)


def _validate_input_dtype(name: str, actual: np.dtype[Any], expected: np.dtype[Any]) -> None:
    if np.dtype(actual) != np.dtype(expected):
        raise TypeError(f"TensorRT input {name} expects {np.dtype(expected)}, got {np.dtype(actual)}")


def _cuda_device_id(device: str) -> int:
    if device == "cuda":
        return 0
    if not device.startswith("cuda:"):
        raise ValueError(f"TensorRT device must be cuda or cuda:<index>, got: {device}")
    raw_index = device.split(":", 1)[1]
    try:
        index = int(raw_index)
    except ValueError as exc:
        raise ValueError(f"invalid CUDA device index: {device}") from exc
    if index < 0:
        raise ValueError(f"invalid CUDA device index: {device}")
    return index


def _resolved_shape(name: str, values: Any) -> tuple[int, ...]:
    shape = tuple(int(value) for value in values)
    if any(value < 0 for value in shape):
        raise RuntimeError(f"unresolved TensorRT output shape for {name}: {shape}")
    return shape


def _tensor_nbytes(shape: tuple[int, ...], dtype: np.dtype[Any]) -> int:
    elements = int(np.prod(shape, dtype=np.int64)) if shape else 1
    nbytes = elements * int(np.dtype(dtype).itemsize)
    if nbytes <= 0:
        raise RuntimeError(f"TensorRT tensor has an empty allocation: shape={shape}, dtype={dtype}")
    return nbytes


def _release_buffers(buffers: dict[str, _TensorBuffer]) -> None:
    first_error: Exception | None = None
    for buffer in buffers.values():
        try:
            buffer.device.close()
        except Exception as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


def _require_trt_success(result: Any, operation: str) -> None:
    if result is not True:
        raise RuntimeError(f"TensorRT {operation} failed")
