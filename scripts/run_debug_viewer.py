from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_deploy.camera.realsense_source import RealSenseSource
from visual_deploy.config import load_config, resolve_path
from visual_deploy.debug.overlay import DebugOverlayOptions, depth_to_bgr, render_debug_overlay
from visual_deploy.debug.snapshot import DebugSnapshot
from visual_deploy.detection.yolo_detector import MockDetector, UltralyticsYoloDetector
from visual_deploy.pipeline.offline_pipeline import OfflinePipeline
from visual_deploy.segmentation.gcnet_segmentor import GCNetSegmentor, SegmentResult
from visual_deploy.types import CameraIntrinsics, DeployFrame, GraspTarget


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

    detector = _build_detector(config_path, config, use_mock_models=bool(args.use_mock_models))
    segmentor = _build_segmentor(config_path, config, use_mock_models=bool(args.use_mock_models))
    pipeline = OfflinePipeline(detector=detector, segmentor=segmentor, config=config)
    source = _build_source(args, config)

    paused = False
    show_masks = True
    show_depth = False
    last_frame: DeployFrame | None = None
    last_snapshot: DebugSnapshot | None = None
    last_target: GraspTarget | None = None
    last_fps = 0.0
    previous_time = time.perf_counter()

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    try:
        for idx, frame in enumerate(source):
            if not paused:
                last_frame = frame
                last_target = pipeline.process_frame(frame)
                last_snapshot = pipeline.last_debug
                now = time.perf_counter()
                elapsed = max(now - previous_time, 1e-6)
                last_fps = 1.0 / elapsed
                previous_time = now
                print(json.dumps(asdict(last_target), ensure_ascii=False))

            if last_frame is None:
                continue
            view = _render_view(
                last_frame,
                last_snapshot,
                last_target,
                fps=last_fps,
                paused=paused,
                show_masks=show_masks,
                show_depth=show_depth,
                show_labels=bool(args.labels),
            )
            cv2.imshow(args.window_name, view)

            key = cv2.waitKey(int(args.wait_ms)) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("p"):
                paused = not paused
            elif key == ord("m"):
                show_masks = not show_masks
            elif key == ord("d"):
                show_depth = not show_depth
            elif key == ord("s") and last_frame is not None:
                _save_snapshot(pipeline.recorder.run_dir, last_frame, view)
            if args.max_frames is not None and idx + 1 >= args.max_frames:
                break
    finally:
        if hasattr(source, "close"):
            source.close()
        cv2.destroyAllWindows()


def _render_view(
    frame: DeployFrame,
    snapshot: DebugSnapshot | None,
    target: GraspTarget | None,
    *,
    fps: float,
    paused: bool,
    show_masks: bool,
    show_depth: bool,
    show_labels: bool,
) -> np.ndarray:
    detections = snapshot.detections if snapshot is not None else []
    masks = snapshot.masks if snapshot is not None and show_masks else []
    lines = _viewer_status_lines(frame_id=frame.frame_id, fps=fps, target=target, paused=paused)
    overlay = render_debug_overlay(
        frame.color_bgr,
        detections=detections,
        target=target,
        masks=masks,
        options=DebugOverlayOptions(show_labels=show_labels, show_status=False),
    )
    if show_depth:
        depth_preview = depth_to_bgr(frame.depth_mm)
        if depth_preview.shape[:2] != overlay.shape[:2]:
            depth_preview = cv2.resize(depth_preview, (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST)
        overlay = np.hstack([overlay, depth_preview])
    return _append_status_bar(overlay, lines)


def _viewer_status_lines(frame_id: int, fps: float, target: GraspTarget | None, paused: bool) -> list[str]:
    state = "paused" if paused else "live"
    lines = [f"frame={frame_id} fps={fps:.1f} {state}"]
    if target is None:
        return lines
    if not target.valid:
        reason = target.reason or "invalid"
        lines.append(f"target invalid: {reason}")
        return lines

    track = "-" if target.track_id is None else str(target.track_id)
    u = 0.0 if target.u_px is None else target.u_px
    v = 0.0 if target.v_px is None else target.v_px
    z = 0.0 if target.z_mm is None else target.z_mm
    score = 0.0 if target.target_score is None else target.target_score
    lines.append(f"target id={track} u={u:.1f} v={v:.1f} z={z:.0f}mm score={score:.2f}")
    lines.append("keys: q quit | p pause | m mask | d depth | s save")
    return lines


def _append_status_bar(image: np.ndarray, lines: list[str]) -> np.ndarray:
    line_h = 18
    pad = 7
    height = pad * 2 + line_h * min(len(lines), 4)
    bar = np.zeros((height, image.shape[1], 3), dtype=np.uint8)
    for idx, line in enumerate(lines[:4]):
        cv2.putText(
            bar,
            str(line)[:100],
            (pad, pad + 13 + idx * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
    return np.vstack([image, bar])


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


def _build_source(args: argparse.Namespace, config: dict) -> Iterator[DeployFrame]:
    if args.rgb is not None or args.depth is not None:
        return _offline_source(args)

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


def _offline_source(args: argparse.Namespace) -> Iterator[DeployFrame]:
    if args.rgb is None or args.depth is None:
        raise ValueError("--rgb and --depth must be provided together for offline debug mode")
    for name in ("fx", "fy", "ppx", "ppy"):
        if getattr(args, name) is None:
            raise ValueError(f"--{name} is required for offline debug mode")

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
    repeat = max(1, int(args.repeat_frames))
    for idx in range(repeat):
        yield DeployFrame(
            frame_id=idx,
            timestamp_ms=float(idx),
            color_bgr=color_bgr.copy(),
            depth_mm=depth_mm.copy(),
            intrinsics=intrinsics,
            meta={"source": "offline_debug"},
        )


def _save_snapshot(run_dir: Path, frame: DeployFrame, view_bgr: np.ndarray) -> None:
    debug_dir = run_dir / "debug_snapshots"
    debug_dir.mkdir(exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{frame.frame_id:06d}_view.jpg"), view_bgr)
    cv2.imwrite(str(debug_dir / f"{frame.frame_id:06d}_rgb.jpg"), frame.color_bgr)
    np.save(debug_dir / f"{frame.frame_id:06d}_depth_mm.npy", frame.depth_mm)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight RGB-D deployment debug viewer.")
    parser.add_argument("--config", default="configs/deploy.yaml")
    parser.add_argument("--window-name", default="Visual-Deploy Debug")
    parser.add_argument("--wait-ms", type=int, default=1)
    parser.add_argument("--labels", action="store_true", help="draw small bbox/target labels on the image")
    parser.add_argument("--use-mock-models", action="store_true", help="allow mock detector/segmentor when weights are missing")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--rgb", type=Path, default=None, help="offline RGB image")
    parser.add_argument("--depth", type=Path, default=None, help="offline depth .npy in millimeters")
    parser.add_argument("--fx", type=float, default=None)
    parser.add_argument("--fy", type=float, default=None)
    parser.add_argument("--ppx", type=float, default=None)
    parser.add_argument("--ppy", type=float, default=None)
    parser.add_argument("--repeat-frames", type=int, default=2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
