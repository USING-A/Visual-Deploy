from __future__ import annotations

from pathlib import Path
from typing import Any

from visual_deploy.config import resolve_path
from visual_deploy.detection.yolo_detector import YoloV10Detector
from visual_deploy.segmentation.gcnet_segmentor import GCNetSegmentor


def build_detector(config_path: Path, config: dict[str, Any]) -> YoloV10Detector:
    cfg = config.get("detection", {})
    weights = resolve_path(config_path, cfg.get("weights", "weights/yolov10_sam_robust_standard.onnx"))
    if not weights.is_file():
        raise FileNotFoundError(f"YOLOv10 model not found: {weights}")
    names = {int(key): str(value) for key, value in cfg.get("class_names", {0: "apple"}).items()}
    return YoloV10Detector(
        weights,
        conf_threshold=float(cfg.get("conf_threshold", 0.5)),
        device=str(cfg.get("device", "cpu")),
        backend=str(cfg.get("backend", "auto")),
        image_size=int(cfg.get("image_size", 640)),
        class_names=names,
        reuse_buffers=_reuse_buffers(cfg),
    )


def build_segmentor(config_path: Path, config: dict[str, Any]) -> GCNetSegmentor:
    cfg = config.get("segmentation", {})
    model = resolve_path(config_path, cfg.get("weights", "weights/rgbd_gcnet_l03_robustft.onnx"))
    if not model.is_file():
        raise FileNotFoundError(f"GCNet model not found: {model}")
    return GCNetSegmentor(
        model,
        device=str(cfg.get("device", "cpu")),
        backend=str(cfg.get("backend", "auto")),
        threshold=float(cfg.get("threshold", 0.5)),
        reuse_buffers=_reuse_buffers(cfg),
    )


def _reuse_buffers(config: dict[str, Any]) -> bool:
    value = config.get("reuse_buffers", True)
    if not isinstance(value, bool):
        raise ValueError("reuse_buffers must be a boolean")
    return value
