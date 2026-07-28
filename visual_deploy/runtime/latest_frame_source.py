from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from time import perf_counter
from typing import Iterable, Iterator

from visual_deploy.observability.runtime_telemetry import (
    FrameRuntimeTelemetry,
    attach_runtime_telemetry,
    read_runtime_telemetry,
)
from visual_deploy.types import DeployFrame


_QUEUE_NAME = "capture_to_inference"
_ENQUEUED_MONOTONIC_MS = "_latest_frame_enqueued_monotonic_ms"


@dataclass(frozen=True)
class _CaptureFailure:
    error: BaseException


class _EndOfStream:
    pass


_END = _EndOfStream()


class LatestFrameSource(Iterator[DeployFrame]):
    """Capture frames in one thread and retain only the freshest frame.

    The queue capacity is intentionally fixed at one. Stateful perception
    remains in the consumer thread, while stale camera frames are replaced
    instead of accumulating latency.
    """

    def __init__(self, source: Iterable[DeployFrame]) -> None:
        self.source = source
        self._queue: Queue[DeployFrame | _CaptureFailure | _EndOfStream] = Queue(maxsize=1)
        self._stop = Event()
        self._thread: Thread | None = None
        self._dropped_frames = 0
        self._drop_lock = Lock()

    @property
    def dropped_frames(self) -> int:
        with self._drop_lock:
            return self._dropped_frames

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = Thread(target=self._capture_loop, name="rgbd-capture", daemon=True)
        self._thread.start()

    def __iter__(self) -> LatestFrameSource:
        self.start()
        return self

    def __next__(self) -> DeployFrame:
        self.start()
        item = self._queue.get()
        if item is _END:
            raise StopIteration
        if isinstance(item, _CaptureFailure):
            raise item.error

        frame = item
        dequeued_ms = perf_counter() * 1000.0
        enqueued_ms = float(frame.meta.pop(_ENQUEUED_MONOTONIC_MS, dequeued_ms))
        current = read_runtime_telemetry(frame)
        attach_runtime_telemetry(
            frame,
            FrameRuntimeTelemetry(
                capture_ms=current.capture_ms,
                captured_monotonic_ms=current.captured_monotonic_ms,
                queue_wait_ms={**current.queue_wait_ms, _QUEUE_NAME: max(0.0, dequeued_ms - enqueued_ms)},
                queue_depth={**current.queue_depth, _QUEUE_NAME: self._queue.qsize()},
                queue_capacity={**current.queue_capacity, _QUEUE_NAME: 1},
                dropped_frames={**current.dropped_frames, _QUEUE_NAME: self.dropped_frames},
                resources=current.resources,
            ),
        )
        return frame

    def close(self, timeout_s: float = 2.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.0, float(timeout_s)))

    def _capture_loop(self) -> None:
        try:
            for frame in self.source:
                if self._stop.is_set():
                    break
                self._offer_latest(frame)
        except BaseException as exc:
            self._put_control(_CaptureFailure(exc))
            return
        self._put_control(_END)

    def _offer_latest(self, frame: DeployFrame) -> None:
        while True:
            frame.meta[_ENQUEUED_MONOTONIC_MS] = perf_counter() * 1000.0
            try:
                self._queue.put_nowait(frame)
                return
            except Full:
                try:
                    removed = self._queue.get_nowait()
                except Empty:
                    continue
                if isinstance(removed, DeployFrame):
                    with self._drop_lock:
                        self._dropped_frames += 1

    def _put_control(self, item: _CaptureFailure | _EndOfStream) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return
            except Full:
                continue
