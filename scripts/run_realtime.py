from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_deploy.camera.realsense_source import RealSenseSource
from visual_deploy.config import load_config, resolve_path
from visual_deploy.detection.yolo_detector import UltralyticsYoloDetector
from visual_deploy.pipeline.offline_pipeline import OfflinePipeline
from visual_deploy.segmentation.gcnet_segmentor import GCNetSegmentor


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    detector = _build_detector(config_path, config)
    segmentor = _build_segmentor(config_path, config)
    source = _build_source(config)
    pipeline = OfflinePipeline(detector=detector, segmentor=segmentor, config=config)

    try:
        for idx, frame in enumerate(source):
            target = pipeline.process_frame(frame)
            print(json.dumps(asdict(target), ensure_ascii=False))
            if args.max_frames is not None and idx + 1 >= args.max_frames:
                break
    finally:
        source.close()


def _build_detector(config_path: Path, config: dict) -> UltralyticsYoloDetector:
    det_cfg = config.get("detection", {})
    weights = resolve_path(config_path, det_cfg.get("weights", "weights/yolo_detect.pt"))
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    return UltralyticsYoloDetector(
        weights,
        conf_threshold=float(det_cfg.get("conf_threshold", 0.25)),
        device=det_cfg.get("device"),
        model_type=det_cfg.get("model_type", "auto"),
    )


def _build_segmentor(config_path: Path, config: dict) -> GCNetSegmentor:
    seg_cfg = config.get("segmentation", {})
    weights = resolve_path(config_path, seg_cfg.get("weights", "weights/rgbd_gcnet_l03_robustft_inference.pt"))
    if not weights.exists():
        raise FileNotFoundError(f"GCNet weights not found: {weights}")
    return GCNetSegmentor(weights, device=seg_cfg.get("device", "cpu"))


def _build_source(config: dict) -> RealSenseSource:
    camera = config.get("camera", {})
    return RealSenseSource(
        width=camera.get("width", 640),
        height=camera.get("height", 480),
        fps=camera.get("fps", 30),
        align_to_color=bool(camera.get("align_to_color", True)),
        spatial_filter=bool(camera.get("spatial_filter", True)),
        temporal_filter=bool(camera.get("temporal_filter", False)),
        hole_filling_filter=bool(camera.get("hole_filling_filter", False)),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synchronous RealSense RGB-D grasp deployment.")
    parser.add_argument("--config", default="configs/deploy.yaml")
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
