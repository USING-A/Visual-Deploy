from __future__ import annotations

from os import PathLike
from typing import Any

import cv2
import numpy as np

from visual_deploy.inference.session import InferenceSession, create_inference_session
from visual_deploy.types import Detection


class MockDetector:
    def __init__(self, confidence: float = 0.9) -> None:
        self.confidence = float(confidence)

    def infer(self, color_bgr: np.ndarray) -> list[Detection]:
        image = np.asarray(color_bgr)
        _validate_color_bgr(image)
        height, width = image.shape[:2]
        return [Detection((width * 0.25, height * 0.25, width * 0.75, height * 0.75), 0, self.confidence, "mock")]


class YoloV10Detector:
    """Standard YOLOv10 end-to-end ONNX/TensorRT detector adapter."""

    def __init__(
        self,
        weights_path: str | PathLike[str],
        conf_threshold: float = 0.5,
        device: str = "cpu",
        backend: str = "auto",
        image_size: int = 640,
        class_names: dict[int, str] | None = None,
        *,
        session: InferenceSession | None = None,
    ) -> None:
        self.conf_threshold = _validate_conf_threshold(conf_threshold)
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        self.class_names = class_names or {0: "apple"}
        self.session = session or create_inference_session(weights_path, device=device, backend=backend)
        if len(self.session.input_names) != 1:
            raise ValueError(f"YOLOv10 model must have one input, got {self.session.input_names}")

    def infer(self, color_bgr: np.ndarray) -> list[Detection]:
        image = np.asarray(color_bgr)
        _validate_color_bgr(image)
        tensor, scale, pad_x, pad_y = _preprocess(image, self.image_size)
        outputs = self.session.run({self.session.input_names[0]: tensor})
        raw = np.asarray(outputs[self.session.output_names[0]])
        if raw.ndim != 3 or raw.shape[0] != 1 or raw.shape[2] != 6:
            raise ValueError(f"YOLOv10 output must have shape (1, N, 6), got {raw.shape}")
        height, width = image.shape[:2]
        detections: list[Detection] = []
        for x1, y1, x2, y2, confidence, class_id_raw in raw[0]:
            confidence = float(confidence)
            if not np.isfinite(confidence) or confidence < self.conf_threshold:
                continue
            box = np.array([(x1 - pad_x) / scale, (y1 - pad_y) / scale, (x2 - pad_x) / scale, (y2 - pad_y) / scale])
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if not np.isfinite(box).all() or box[2] <= box[0] or box[3] <= box[1]:
                continue
            class_id = int(class_id_raw)
            detections.append(
                Detection(tuple(float(value) for value in box), class_id, confidence, self.class_names.get(class_id, ""))
            )
        return detections


def _preprocess(image: np.ndarray, image_size: int) -> tuple[np.ndarray, float, float, float]:
    height, width = image.shape[:2]
    scale = min(image_size / height, image_size / width)
    resized_width, resized_height = round(width * scale), round(height * scale)
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (image_size - resized_width) / 2.0
    pad_y = (image_size - resized_height) / 2.0
    left, right = round(pad_x - 0.1), round(pad_x + 0.1)
    top, bottom = round(pad_y - 0.1), round(pad_y + 0.1)
    letterboxed = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    rgb = letterboxed[:, :, ::-1].transpose(2, 0, 1)
    tensor = np.ascontiguousarray(rgb[None], dtype=np.float32) / 255.0
    return tensor, scale, float(left), float(top)


def _validate_color_bgr(color_bgr: np.ndarray) -> None:
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must have shape (height, width, 3)")
    if not np.isfinite(color_bgr.astype(np.float32, copy=False)).all():
        raise ValueError("color_bgr must contain only finite values")


def _validate_conf_threshold(conf_threshold: float) -> float:
    try:
        value = float(conf_threshold)
    except (TypeError, ValueError) as exc:
        raise ValueError("conf_threshold must be a finite float in [0.0, 1.0]") from exc
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("conf_threshold must be a finite float in [0.0, 1.0]")
    return value
