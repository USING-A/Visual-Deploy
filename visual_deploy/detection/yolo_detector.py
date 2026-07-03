from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from visual_deploy.types import Detection

try:
    from ultralytics import YOLO
    try:
        from ultralytics import YOLOv10
    except Exception:
        YOLOv10 = None
    _YOLO_IMPORT_ERROR = None
except Exception as exc:
    YOLO = None
    YOLOv10 = None
    _YOLO_IMPORT_ERROR = exc


class MockDetector:
    def __init__(self, confidence: float = 0.9) -> None:
        self.confidence = float(confidence)

    def infer(self, color_bgr: np.ndarray) -> list[Detection]:
        image = np.asarray(color_bgr)
        _validate_color_bgr(image)
        height, width = image.shape[:2]
        return [
            Detection(
                bbox_xyxy=(width * 0.25, height * 0.25, width * 0.75, height * 0.75),
                class_id=0,
                confidence=self.confidence,
                label="mock",
            )
        ]


class UltralyticsYoloDetector:
    def __init__(
        self,
        weights_path: str | PathLike[str],
        conf_threshold: float = 0.25,
        device: str | None = None,
        model_type: str = "auto",
    ) -> None:
        weights = Path(weights_path)
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        conf_threshold = _validate_conf_threshold(conf_threshold)
        self.model_type = _validate_model_type(model_type)
        model_class = _select_model_class(self.model_type)

        self.weights_path = weights
        self.conf_threshold = conf_threshold
        self.device = device
        self.model = model_class(str(weights))

    def infer(self, color_bgr: np.ndarray) -> list[Detection]:
        image = np.asarray(color_bgr)
        _validate_color_bgr(image)
        results = self.model.predict(
            source=color_bgr,
            conf=self.conf_threshold,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []

        names = getattr(result, "names", {}) or {}
        detections: list[Detection] = []
        for box in boxes:
            class_id = int(_scalar(getattr(box, "cls")))
            detections.append(
                Detection(
                    bbox_xyxy=tuple(float(value) for value in _xyxy_values(getattr(box, "xyxy"))),
                    class_id=class_id,
                    confidence=float(_scalar(getattr(box, "conf"))),
                    label=str(names.get(class_id, "")) if hasattr(names, "get") else "",
                )
            )
        return detections


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


def _validate_model_type(model_type: str) -> str:
    value = str(model_type).lower()
    if value not in {"auto", "yolo", "yolov10"}:
        raise ValueError("model_type must be one of: auto, yolo, yolov10")
    return value


def _select_model_class(model_type: str):
    if model_type == "yolov10":
        if YOLOv10 is None:
            raise ImportError("ultralytics.YOLOv10 is required for model_type='yolov10'")
        return YOLOv10
    if model_type == "yolo":
        if YOLO is None:
            message = "ultralytics.YOLO is required for model_type='yolo'"
            if _YOLO_IMPORT_ERROR is not None:
                message = f"{message}: {_YOLO_IMPORT_ERROR}"
            raise ImportError(message) from _YOLO_IMPORT_ERROR
        return YOLO
    model_class = YOLOv10 or YOLO
    if model_class is None:
        message = "ultralytics is required to use UltralyticsYoloDetector"
        if _YOLO_IMPORT_ERROR is not None:
            message = f"{message}: {_YOLO_IMPORT_ERROR}"
        raise ImportError(message) from _YOLO_IMPORT_ERROR
    return model_class


def _xyxy_values(value: Any) -> list[float]:
    first = value[0] if hasattr(value, "__getitem__") else value
    values = first.tolist() if hasattr(first, "tolist") else first
    if len(values) != 4:
        raise ValueError("YOLO box xyxy must contain four values")
    return values


def _scalar(value: Any) -> float:
    first = value[0] if hasattr(value, "__getitem__") else value
    return first.item() if hasattr(first, "item") else first
