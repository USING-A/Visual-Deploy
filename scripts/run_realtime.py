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
from visual_deploy.config import load_config
from visual_deploy.inference.factory import build_detector, build_segmentor
from visual_deploy.pipeline.offline_pipeline import OfflinePipeline


def main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)

    detector = build_detector(config_path, config)
    segmentor = build_segmentor(config_path, config)
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
