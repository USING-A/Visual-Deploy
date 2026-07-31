from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
from typing import Any


class CudaRuntimeError(RuntimeError):
    """A CUDA Runtime API call failed."""


class CudaRuntime:
    """Minimal owner for the CUDA Runtime calls required by TensorRT inference."""

    MEMCPY_HOST_TO_DEVICE = 1
    MEMCPY_DEVICE_TO_HOST = 2
    STREAM_NON_BLOCKING = 1

    def __init__(self, device_id: int = 0, *, library: Any | None = None) -> None:
        if not isinstance(device_id, int) or isinstance(device_id, bool) or device_id < 0:
            raise ValueError("CUDA device id must be a non-negative integer")
        self._library = library if library is not None else _load_cudart()
        self._configure_signatures()

        count = ctypes.c_int()
        self._check(self._library.cudaGetDeviceCount(ctypes.byref(count)), "cudaGetDeviceCount")
        if device_id >= count.value:
            raise RuntimeError(f"CUDA device {device_id} is unavailable; detected {count.value} device(s)")
        self._check(self._library.cudaSetDevice(device_id), f"cudaSetDevice({device_id})")
        self.device_id = device_id

    def allocate(self, nbytes: int) -> CudaDeviceBuffer:
        if not isinstance(nbytes, int) or isinstance(nbytes, bool) or nbytes <= 0:
            raise ValueError("CUDA allocation size must be a positive integer")
        pointer = ctypes.c_void_p()
        self._check(self._library.cudaMalloc(ctypes.byref(pointer), nbytes), f"cudaMalloc({nbytes})")
        if not pointer.value:
            raise CudaRuntimeError(f"cudaMalloc({nbytes}) returned a null pointer")
        return CudaDeviceBuffer(self, int(pointer.value), nbytes)

    def create_stream(self) -> CudaStream:
        pointer = ctypes.c_void_p()
        self._check(
            self._library.cudaStreamCreateWithFlags(ctypes.byref(pointer), self.STREAM_NON_BLOCKING),
            "cudaStreamCreateWithFlags",
        )
        if not pointer.value:
            raise CudaRuntimeError("cudaStreamCreateWithFlags returned a null stream")
        return CudaStream(self, int(pointer.value))

    def copy_host_to_device_async(
        self,
        destination: CudaDeviceBuffer,
        source: Any,
        stream: CudaStream,
    ) -> None:
        nbytes = int(source.nbytes)
        _validate_copy_size(destination, nbytes)
        if nbytes == 0:
            return
        self._memcpy_async(destination.pointer, int(source.ctypes.data), nbytes, self.MEMCPY_HOST_TO_DEVICE, stream)

    def copy_device_to_host_async(
        self,
        destination: Any,
        source: CudaDeviceBuffer,
        stream: CudaStream,
    ) -> None:
        nbytes = int(destination.nbytes)
        _validate_copy_size(source, nbytes)
        if nbytes == 0:
            return
        self._memcpy_async(int(destination.ctypes.data), source.pointer, nbytes, self.MEMCPY_DEVICE_TO_HOST, stream)

    def _memcpy_async(self, destination: int, source: int, nbytes: int, kind: int, stream: CudaStream) -> None:
        if stream.closed:
            raise RuntimeError("CUDA stream is closed")
        self._check(
            self._library.cudaMemcpyAsync(
                ctypes.c_void_p(destination),
                ctypes.c_void_p(source),
                nbytes,
                kind,
                ctypes.c_void_p(stream.handle),
            ),
            "cudaMemcpyAsync",
        )

    def _free(self, pointer: int) -> None:
        self._check(self._library.cudaFree(ctypes.c_void_p(pointer)), "cudaFree")

    def _synchronize_stream(self, handle: int) -> None:
        self._check(self._library.cudaStreamSynchronize(ctypes.c_void_p(handle)), "cudaStreamSynchronize")

    def _destroy_stream(self, handle: int) -> None:
        self._check(self._library.cudaStreamDestroy(ctypes.c_void_p(handle)), "cudaStreamDestroy")

    def _check(self, result: Any, operation: str) -> None:
        code = int(result)
        if code == 0:
            return
        message = self._library.cudaGetErrorString(code)
        if isinstance(message, bytes):
            detail = message.decode("utf-8", errors="replace")
        else:
            detail = str(message or "unknown CUDA error")
        raise CudaRuntimeError(f"{operation} failed with CUDA error {code}: {detail}")

    def _configure_signatures(self) -> None:
        signatures = {
            "cudaGetDeviceCount": ([ctypes.POINTER(ctypes.c_int)], ctypes.c_int),
            "cudaSetDevice": ([ctypes.c_int], ctypes.c_int),
            "cudaMalloc": ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t], ctypes.c_int),
            "cudaFree": ([ctypes.c_void_p], ctypes.c_int),
            "cudaMemcpyAsync": (
                [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p],
                ctypes.c_int,
            ),
            "cudaStreamCreateWithFlags": ([ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint], ctypes.c_int),
            "cudaStreamSynchronize": ([ctypes.c_void_p], ctypes.c_int),
            "cudaStreamDestroy": ([ctypes.c_void_p], ctypes.c_int),
            "cudaGetErrorString": ([ctypes.c_int], ctypes.c_char_p),
        }
        for name, (argtypes, restype) in signatures.items():
            function = getattr(self._library, name, None)
            if function is None:
                raise ImportError(f"CUDA Runtime library does not provide {name}")
            function.argtypes = argtypes
            function.restype = restype


@dataclass
class CudaDeviceBuffer:
    runtime: CudaRuntime
    pointer: int
    nbytes: int

    @property
    def closed(self) -> bool:
        return self.pointer == 0

    def close(self) -> None:
        if self.closed:
            return
        pointer = self.pointer
        self.runtime._free(pointer)
        self.pointer = 0


@dataclass
class CudaStream:
    runtime: CudaRuntime
    handle: int

    @property
    def closed(self) -> bool:
        return self.handle == 0

    def synchronize(self) -> None:
        if self.closed:
            raise RuntimeError("CUDA stream is closed")
        self.runtime._synchronize_stream(self.handle)

    def close(self) -> None:
        if self.closed:
            return
        handle = self.handle
        self.runtime._destroy_stream(handle)
        self.handle = 0


def _load_cudart() -> Any:
    candidates = []
    discovered = ctypes.util.find_library("cudart")
    if discovered:
        candidates.append(discovered)
    candidates.append("libcudart.so")

    errors = []
    for name in dict.fromkeys(candidates):
        try:
            return ctypes.CDLL(name)
        except OSError as exc:
            errors.append(f"{name}: {exc}")
    raise ImportError(
        "TensorRT engine inference requires JetPack's CUDA Runtime library (libcudart); "
        + "; ".join(errors)
    )


def _validate_copy_size(buffer: CudaDeviceBuffer, nbytes: int) -> None:
    if buffer.closed:
        raise RuntimeError("CUDA device buffer is closed")
    if nbytes > buffer.nbytes:
        raise ValueError(f"copy needs {nbytes} bytes but CUDA buffer has {buffer.nbytes}")
