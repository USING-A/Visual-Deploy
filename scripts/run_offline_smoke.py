from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_deploy.config import load_config, resolve_path
from visual_deploy.detection.yolo_detector import MockDetector, UltralyticsYoloDetector
from visual_deploy.pipeline.offline_pipeline import OfflinePipeline
from visual_deploy.segmentation.gcnet_segmentor import GCNetSegmentor, SegmentResult
from visual_deploy.types import CameraIntrinsics, DeployFrame


class _MockSegmentor:
    def infer(self, color_bgr_256: np.ndarray, depth_mm_256: np.ndarray) -> SegmentResult:
        mask = np.zeros((256, 256), dtype=bool)
        yy, xx = np.ogrid[:256, :256]
        mask[(yy - 128) ** 2 + (xx - 128) ** 2 <= 60 ** 2] = True
        return SegmentResult(mask.astype(np.float32), mask, int(mask.sum()), 1.0)


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    color_bgr = cv2.imread(str(args.rgb), cv2.IMREAD_COLOR)
    if color_bgr is None:
        raise FileNotFoundError(f"failed to read RGB image: {args.rgb}")
    depth_mm = np.load(args.depth).astype(np.float32, copy=False)

    intrinsics = CameraIntrinsics(
        fx=float(args.fx),
        fy=float(args.fy),
        ppx=float(args.ppx),
        ppy=float(args.ppy),
        depth_scale=0.001,
    )
    frame = DeployFrame(
        frame_id=0,
        timestamp_ms=0.0,
        color_bgr=color_bgr,
        depth_mm=depth_mm,
        intrinsics=intrinsics,
        meta={"source": "offline_smoke"},
    )

    detector = _build_detector(config_path, config, use_mock_models=bool(args.use_mock_models))
    segmentor = _build_segmentor(config_path, config, use_mock_models=bool(args.use_mock_models))
    pipeline = OfflinePipeline(detector=detector, segmentor=segmentor, config=config)
    target = None
    for idx in range(max(1, int(args.repeat_frames))):
        frame.frame_id = idx
        frame.timestamp_ms = float(idx)
        target = pipeline.process_frame(frame)
    assert target is not None
    print(json.dumps(asdict(target), ensure_ascii=False, indent=2))


def _build_detector(config_path: Path, config: dict, *, use_mock_models: bool) -> object:
    det_cfg = config.get("detection", {})
    weights = resolve_path(config_path, det_cfg.get("weights", "weights/yolo_detect.pt"))
    if weights.exists():
        return UltralyticsYoloDetector(
            weights,
            conf_threshold=float(det_cfg.get("conf_threshold", 0.25)),
            device=det_cfg.get("device"),
            model_type=det_cfg.get("model_type", "auto"),
        )
    if not use_mock_models:
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    return MockDetector(confidence=0.9)


def _build_segmentor(config_path: Path, config: dict, *, use_mock_models: bool) -> object:
    seg_cfg = config.get("segmentation", {})
    weights = resolve_path(config_path, seg_cfg.get("weights", "weights/rgbd_gcnet_l03_robustft_inference.pt"))
    if weights.exists():
        return GCNetSegmentor(weights, device=seg_cfg.get("device", "cpu"))
    if not use_mock_models:
        raise FileNotFoundError(f"GCNet weights not found: {weights}")
    return _MockSegmentor()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one offline RGB-D grasp smoke frame.")
    parser.add_argument("--config", default="configs/deploy.yaml")
    parser.add_argument("--rgb", required=True, type=Path)
    parser.add_argument("--depth", required=True, type=Path)
    parser.add_argument("--fx", required=True, type=float)
    parser.add_argument("--fy", required=True, type=float)
    parser.add_argument("--ppx", required=True, type=float)
    parser.add_argument("--ppy", required=True, type=float)
    parser.add_argument("--repeat-frames", type=int, default=2)
    parser.add_argument("--use-mock-models", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
