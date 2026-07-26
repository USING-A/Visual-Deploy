from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any, Protocol

import numpy as np


class InferenceSession(Protocol):
    @property
    def input_names(self) -> tuple[str, ...]: ...

    @property
    def output_names(self) -> tuple[str, ...]: ...

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]: ...


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
        self._input_names = tuple(item.name for item in self.session.get_inputs())
        self._output_names = tuple(item.name for item in self.session.get_outputs())

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        _validate_input_names(inputs, self.input_names)
        arrays = {name: np.ascontiguousarray(inputs[name]) for name in self.input_names}
        outputs = self.session.run(list(self.output_names), arrays)
        return dict(zip(self.output_names, outputs))


class TensorRTEngineSession:
    """Load a raw TensorRT engine produced by ``trtexec``.

    Both the TensorRT 8 binding API and TensorRT 10 tensor API are supported.
    CUDA buffers are backed by PyTorch so no additional PyCUDA dependency is
    required on Jetson.
    """

    def __init__(self, model_path: str | PathLike[str], device: str = "cuda:0") -> None:
        path = _validate_model_path(model_path, ".engine")
        try:
            import tensorrt as trt
            import torch
        except ImportError as exc:
            raise ImportError("TensorRT engine inference requires tensorrt and CUDA-enabled torch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT engine inference requires an available CUDA device")

        self.trt = trt
        self.torch = torch
        self.device = torch.device(device)
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

    @property
    def input_names(self) -> tuple[str, ...]:
        return self._input_names

    @property
    def output_names(self) -> tuple[str, ...]:
        return self._output_names

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        _validate_input_names(inputs, self.input_names)
        if self._tensor_api:
            return self._run_tensor_api(inputs)
        return self._run_binding_api(inputs)

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

    def _run_tensor_api(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch = self.torch
        buffers: dict[str, Any] = {}
        for name in self.input_names:
            tensor = torch.as_tensor(inputs[name], device=self.device).contiguous()
            if not self.context.set_input_shape(name, tuple(tensor.shape)):
                raise ValueError(f"TensorRT rejected input shape for {name}: {tuple(tensor.shape)}")
            buffers[name] = tensor
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if any(int(value) < 0 for value in shape):
                raise RuntimeError(f"unresolved TensorRT output shape for {name}: {shape}")
            dtype = _torch_dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)), torch)
            buffers[name] = torch.empty(shape, dtype=dtype, device=self.device)
        for name, tensor in buffers.items():
            _require_trt_success(
                self.context.set_tensor_address(name, int(tensor.data_ptr())),
                f"set_tensor_address({name})",
            )
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v3(stream_handle=stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        stream.synchronize()
        return {name: buffers[name].detach().cpu().numpy() for name in self.output_names}

    def _run_binding_api(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        torch = self.torch
        buffers: dict[str, Any] = {}
        addresses = [0] * self.engine.num_bindings
        for name in self.input_names:
            index = self.engine.get_binding_index(name)
            tensor = torch.as_tensor(inputs[name], device=self.device).contiguous()
            if -1 in tuple(self.engine.get_binding_shape(index)):
                _require_trt_success(
                    self.context.set_binding_shape(index, tuple(tensor.shape)),
                    f"set_binding_shape({name})",
                )
            buffers[name] = tensor
            addresses[index] = int(tensor.data_ptr())
        for name in self.output_names:
            index = self.engine.get_binding_index(name)
            shape = tuple(self.context.get_binding_shape(index))
            if any(int(value) < 0 for value in shape):
                raise RuntimeError(f"unresolved TensorRT output shape for {name}: {shape}")
            dtype = _torch_dtype(self.trt.nptype(self.engine.get_binding_dtype(index)), torch)
            tensor = torch.empty(shape, dtype=dtype, device=self.device)
            buffers[name] = tensor
            addresses[index] = int(tensor.data_ptr())
        stream = torch.cuda.current_stream(self.device)
        if not self.context.execute_async_v2(addresses, stream.cuda_stream):
            raise RuntimeError("TensorRT execute_async_v2 failed")
        stream.synchronize()
        return {name: buffers[name].detach().cpu().numpy() for name in self.output_names}


def create_inference_session(
    model_path: str | PathLike[str],
    *,
    device: str = "cpu",
    backend: str = "auto",
) -> InferenceSession:
    path = Path(model_path)
    selected = str(backend).lower()
    if selected == "auto":
        selected = {".onnx": "onnxruntime", ".engine": "tensorrt"}.get(path.suffix.lower(), "")
    if selected == "onnxruntime":
        return OnnxRuntimeSession(path, device=device)
    if selected == "tensorrt":
        return TensorRTEngineSession(path, device=device)
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


def _cuda_device_id(device: str) -> int:
    parts = device.split(":", 1)
    return int(parts[1]) if len(parts) == 2 and parts[1] else 0


def _torch_dtype(numpy_dtype: Any, torch: Any) -> Any:
    mapping = {
        np.dtype(np.float16): torch.float16,
        np.dtype(np.float32): torch.float32,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int64): torch.int64,
        np.dtype(np.uint8): torch.uint8,
        np.dtype(np.bool_): torch.bool,
    }
    dtype = mapping.get(np.dtype(numpy_dtype))
    if dtype is None:
        raise TypeError(f"unsupported TensorRT tensor dtype: {numpy_dtype}")
    return dtype


def _require_trt_success(result: Any, operation: str) -> None:
    if result is not True:
        raise RuntimeError(f"TensorRT {operation} failed")
