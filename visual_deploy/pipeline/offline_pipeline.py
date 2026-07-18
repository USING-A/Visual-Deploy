from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any

import cv2
import numpy as np

from visual_deploy.debug.overlay import render_debug_overlay
from visual_deploy.debug.snapshot import DebugSnapshot
from visual_deploy.geometry.backproject import backproject_pixel
from visual_deploy.geometry.grasp_patch import GraspPatch, select_grasp_patch
from visual_deploy.geometry.pose import approach_from_normal
from visual_deploy.geometry.roi import RoiTransform
from visual_deploy.ranking.target_ranker import CandidateScores, TargetRanker
from visual_deploy.recording.recorder import RunRecorder
from visual_deploy.safety.target_continuity import TargetContinuityValidator
from visual_deploy.safety.target_validator import TargetSafetyValidator, TargetValidation
from visual_deploy.tracking.depth_fusion import DepthFusionBuffer, FusedTrackDepth
from visual_deploy.tracking.tracker import ConfidenceGate, DepthRoiStats, SimpleIoUTracker
from visual_deploy.types import DeployFrame, Detection, GraspTarget, Track


_TIMING_FIELDS = (
    "detection_ms",
    "tracking_ms",
    "depth_fusion_ms",
    "segmentation_ms",
    "grasp_ms",
    "ranking_safety_ms",
    "recording_ms",
    "diagnostic_ms",
)


class OfflinePipeline:
    def __init__(self, detector: Any, segmentor: Any, config: dict[str, Any] | None = None) -> None:
        self.detector = detector
        self.segmentor = segmentor
        self.config = config or {}

        tracking_cfg = self.config.get("tracking", {})
        self.gate = ConfidenceGate(
            high_threshold=float(tracking_cfg.get("high_conf_threshold", 0.7)),
            low_threshold=float(tracking_cfg.get("low_conf_threshold", 0.4)),
        )
        self.depth_stats = DepthRoiStats(**_pick(self.config.get("depth_fusion", {}), "min_depth_mm", "max_depth_mm"))
        self.min_hits = int(tracking_cfg.get("min_hits", 1))
        self.tracker = SimpleIoUTracker(
            iou_threshold=float(tracking_cfg.get("iou_threshold", 0.3)),
            max_lost=int(tracking_cfg.get("max_lost", 30)),
            min_hits=self.min_hits,
        )
        self.depth_fusion = DepthFusionBuffer(**_pick(self.config.get("depth_fusion", {}), "window_size", "min_depth_mm", "max_depth_mm", "min_valid_ratio", "min_roi_iou"))
        self.ranker = TargetRanker(weights=self.config.get("ranking", {}).get("weights"))
        self.safety_validator = TargetSafetyValidator.from_config(self.config.get("safety"))
        self.continuity_validator = TargetContinuityValidator.from_config(self.config.get("safety"))
        self.last_debug: DebugSnapshot | None = None

        recording_cfg = self.config.get("recording", {})
        self.recorder = RunRecorder(recording_cfg.get("output_root", "runs"), config=self.config)
        self.save_event_artifacts = bool(recording_cfg.get("save_event_artifacts", False))
        self.event_stages = {str(value) for value in recording_cfg.get("event_stages", ("safety", "continuity"))}
        self.event_cooldown_frames = int(recording_cfg.get("event_cooldown_frames", 0))
        if self.event_cooldown_frames < 0:
            raise ValueError("recording.event_cooldown_frames must be non-negative")
        self._last_event_frame_id: int | None = None

    def process_frame(self, frame: DeployFrame) -> GraspTarget:
        frame_started = perf_counter()
        timings = {name: 0.0 for name in _TIMING_FIELDS}
        _validate_frame(frame)
        stage_started = perf_counter()
        detections = self.detector.infer(frame.color_bgr)
        timings["detection_ms"] += _elapsed_ms(stage_started)
        stage_started = perf_counter()
        gated = self.gate.apply(_attach_depth_stats(detections, frame.depth_mm, self.depth_stats))
        tracks = self.tracker.update(gated)
        timings["tracking_ms"] += _elapsed_ms(stage_started)
        stage_started = perf_counter()
        self.recorder.write_detection(
            {
                "frame_id": frame.frame_id,
                "timestamp_ms": frame.timestamp_ms,
                "detections": [_detection_record(detection) for detection in gated],
            }
        )

        candidates: list[_PipelineCandidate] = []
        rejections: list[dict[str, Any]] = []
        for track in tracks:
            if track.state != "confirmed":
                rejections.append(_rejection_record(track, "tracking", "track_not_confirmed"))
                continue
            candidate, rejection = self._process_track(frame, track, timings)
            if candidate is not None:
                candidates.append(candidate)
            if rejection is not None:
                rejections.append(rejection)

        stage_started = perf_counter()
        ranked = self.ranker.rank([candidate.scores for candidate in candidates])
        candidates_by_track = {candidate.track.track_id: candidate for candidate in candidates}
        selected: _PipelineCandidate | None = None
        selected_coordinates: tuple[float, float, float] | None = None
        for rank, score in enumerate(ranked, 1):
            candidate = candidates_by_track[score.track_id]
            candidate.rank = rank
            candidate.validation = self.safety_validator.validate(candidate.scores, candidate.patch)
            coordinates = _candidate_image_coordinates(candidate, frame)
            if candidate.validation.valid:
                candidate.validation = self.continuity_validator.validate(
                    candidate.track.track_id,
                    frame.timestamp_ms,
                    *coordinates,
                )
            if candidate.validation.valid and selected is None:
                selected = candidate
                selected_coordinates = coordinates
                candidate.selected = True
            elif not candidate.validation.valid:
                stage = "continuity" if candidate.validation.reason in {
                    "target_timestamp_regression",
                    "target_depth_jump",
                    "target_pixel_jump",
                    "non_finite_target",
                } else "safety"
                rejections.append(
                    _rejection_record(
                        candidate.track,
                        stage,
                        str(candidate.validation.reason),
                        metric=candidate.validation.metric,
                        value=candidate.validation.value,
                        threshold=candidate.validation.threshold,
                    )
                )
        timings["ranking_safety_ms"] += _elapsed_ms(stage_started)

        stage_started = perf_counter()
        self.recorder.write_candidate(
            {
                "frame_id": frame.frame_id,
                "timestamp_ms": frame.timestamp_ms,
                "candidates": [candidate.to_record() for candidate in candidates],
                "rejections": rejections,
            }
        )
        timings["recording_ms"] += _elapsed_ms(stage_started)
        timings["recording_ms"] += _elapsed_ms(stage_started)

        if selected is None:
            reason = "no_safe_grasp_candidate" if candidates else "no_valid_grasp_candidate"
            target = GraspTarget(valid=False, frame_id=frame.frame_id, reason=reason)
            stage_started = perf_counter()
            self.recorder.write_target(asdict(target))
            timings["recording_ms"] += _elapsed_ms(stage_started)
            self.last_debug = _debug_snapshot(frame, gated, target, candidates)
            self._finalize_frame(frame, gated, candidates, rejections, target, timings, frame_started)
            return target

        best = selected
        best_score = selected.scores
        if selected_coordinates is None:
            raise RuntimeError("selected candidate coordinates are missing")
        u_img, v_img, z_mm = selected_coordinates
        xyz = backproject_pixel(u_img, v_img, z_mm, frame.intrinsics)
        approach = approach_from_normal(best.patch.normal_xyz)
        target = GraspTarget(
            valid=True,
            frame_id=frame.frame_id,
            track_id=best.track.track_id,
            u_px=float(u_img),
            v_px=float(v_img),
            z_mm=float(z_mm),
            xyz_camera_m=tuple(float(value) for value in xyz),
            normal_xyz=tuple(float(value) for value in best.patch.normal_xyz),
            approach_axis=tuple(float(value) for value in approach),
            target_score=float(best_score.target_score),
        )
        self.continuity_validator.accept(best.track.track_id, frame.timestamp_ms, u_img, v_img, z_mm)
        stage_started = perf_counter()
        self.recorder.write_target(asdict(target))
        timings["recording_ms"] += _elapsed_ms(stage_started)
        self.last_debug = _debug_snapshot(frame, gated, target, candidates)
        self._finalize_frame(frame, gated, candidates, rejections, target, timings, frame_started)
        return target

    def _process_track(
        self,
        frame: DeployFrame,
        track: Track,
        timings: dict[str, float],
    ) -> tuple[_PipelineCandidate | None, dict[str, Any] | None]:
        height, width = frame.color_bgr.shape[:2]
        roi_size = 256
        roi = RoiTransform.from_bbox(
            track.bbox_xyxy,
            float(self.config.get("roi", {}).get("pad_ratio", 0.2)),
            width,
            height,
            roi_size,
        )
        color_roi, depth_roi = _crop_resize(frame.color_bgr, frame.depth_mm, roi)
        stage_started = perf_counter()
        fused = self.depth_fusion.update(track.track_id, frame.frame_id, frame.timestamp_ms, roi, depth_roi)
        timings["depth_fusion_ms"] += _elapsed_ms(stage_started)
        if fused.valid_ratio <= 0.0:
            return None, _rejection_record(track, "depth_fusion", "insufficient_depth_fusion")

        stage_started = perf_counter()
        segment = self.segmentor.infer(color_roi, fused.depth_roi_mm)
        timings["segmentation_ms"] += _elapsed_ms(stage_started)
        image_mask = _mask_to_image(segment.mask_256, roi, height, width)
        if segment.mask_area <= 0:
            return None, _rejection_record(
                track,
                "segmentation",
                "empty_mask",
                metric="mask_area",
                value=float(segment.mask_area),
                threshold=1.0,
            )
        stage_started = perf_counter()
        patch = select_grasp_patch(
            segment.mask_256,
            fused.depth_roi_mm,
            roi.adjust_intrinsics(frame.intrinsics),
            **_pick(
                self.config.get("grasp", {}),
                "patch_radius_px",
                "stride_px",
                "min_component_area_px",
                "min_valid_depth_ratio",
                "min_valid_depth_count",
                "max_plane_rmse_m",
                "max_depth_mad_m",
                "min_score",
            ),
        )
        timings["grasp_ms"] += _elapsed_ms(stage_started)
        if patch is None:
            return None, _rejection_record(track, "grasp", "no_valid_grasp_patch")

        scores = CandidateScores(
            track_id=track.track_id,
            grasp_score=float(patch.score),
            depth_valid_score=float(fused.valid_ratio),
            track_confidence=float(track.weighted_confidence if track.weighted_confidence is not None else track.confidence),
            track_stability=min(float(track.hits) / max(float(self.min_hits), 1.0), 1.0),
            mask_quality_score=float(segment.largest_component_ratio),
        )
        return _PipelineCandidate(
            track=track,
            roi_transform=roi,
            depth=fused,
            patch=patch,
            scores=scores,
            image_mask=image_mask,
        ), None

    def _finalize_frame(
        self,
        frame: DeployFrame,
        detections: list[Detection],
        candidates: list[_PipelineCandidate],
        rejections: list[dict[str, Any]],
        target: GraspTarget,
        timings: dict[str, float],
        frame_started: float,
    ) -> None:
        diagnostic_started = perf_counter()
        triggering = [record for record in rejections if record["stage"] in self.event_stages]
        if self._should_save_event(frame.frame_id, triggering):
            recording_cfg = self.config.get("recording", {})
            overlay = None
            if bool(recording_cfg.get("save_overlays", False)):
                overlay = render_debug_overlay(
                    frame.color_bgr,
                    detections=detections,
                    target=target,
                    masks=[candidate.image_mask for candidate in candidates],
                )
            artifacts = self.recorder.save_event_artifacts(
                frame.frame_id,
                color_bgr=frame.color_bgr if bool(recording_cfg.get("save_frames", False)) else None,
                depth_mm=frame.depth_mm if bool(recording_cfg.get("save_depth", False)) else None,
                masks=[(candidate.track.track_id, candidate.image_mask) for candidate in candidates]
                if bool(recording_cfg.get("save_masks", False))
                else None,
                overlay_bgr=overlay,
            )
            self.recorder.write_event(
                {
                    "frame_id": frame.frame_id,
                    "timestamp_ms": frame.timestamp_ms,
                    "target": asdict(target),
                    "rejections": triggering,
                    "artifacts": artifacts,
                }
            )
            self._last_event_frame_id = frame.frame_id
        timings["diagnostic_ms"] += _elapsed_ms(diagnostic_started)
        timings["total_ms"] = _elapsed_ms(frame_started)
        self.recorder.write_timing(
            {
                "frame_id": frame.frame_id,
                "timestamp_ms": frame.timestamp_ms,
                "valid_target": target.valid,
                "target_reason": target.reason,
                **{name: float(value) for name, value in timings.items()},
            }
        )

    def _should_save_event(self, frame_id: int, triggering: list[dict[str, Any]]) -> bool:
        if not self.save_event_artifacts or not triggering:
            return False
        if self._last_event_frame_id is None:
            return True
        return int(frame_id) - self._last_event_frame_id > self.event_cooldown_frames


class _PipelineCandidate:
    def __init__(
        self,
        track: Track,
        roi_transform: RoiTransform,
        depth: FusedTrackDepth,
        patch: GraspPatch,
        scores: CandidateScores,
        image_mask: np.ndarray,
    ) -> None:
        self.track = track
        self.roi_transform = roi_transform
        self.depth = depth
        self.patch = patch
        self.scores = scores
        self.image_mask = image_mask
        self.rank: int | None = None
        self.selected = False
        self.validation = TargetValidation(valid=True)

    def to_record(self) -> dict[str, Any]:
        return {
            "track_id": self.track.track_id,
            "bbox_xyxy": list(self.track.bbox_xyxy),
            "roi_crop_xyxy": list(self.roi_transform.crop_xyxy),
            "rank": self.rank,
            "selected": self.selected,
            "grasp": {
                "u_roi_px": self.patch.u_px,
                "v_roi_px": self.patch.v_px,
                "z_m": self.patch.z_m,
                "score": self.patch.score,
                "valid_depth_ratio": self.patch.valid_depth_ratio,
                "valid_depth_count": self.patch.valid_depth_count,
                "plane_rmse_m": self.patch.plane_rmse_m,
                "depth_mad_m": self.patch.depth_mad_m,
                "normal_xyz": [float(value) for value in self.patch.normal_xyz],
            },
            "depth_fusion": {
                "valid_ratio": self.depth.valid_ratio,
                "source_frame_count": self.depth.source_frame_count,
                "age_ms": self.depth.age_ms,
            },
            "scores": asdict(self.scores),
            "safety": asdict(self.validation),
        }


def _pick(source: dict[str, Any], *names: str) -> dict[str, Any]:
    return {name: source[name] for name in names if name in source}


def _validate_frame(frame: DeployFrame) -> None:
    if frame.color_bgr.ndim != 3 or frame.color_bgr.shape[2] != 3:
        raise ValueError("frame.color_bgr must have shape (height, width, 3)")
    if frame.depth_mm.ndim != 2:
        raise ValueError("frame.depth_mm must be a 2D array")
    if frame.depth_mm.shape != frame.color_bgr.shape[:2]:
        raise ValueError("frame.depth_mm must match color frame height and width")


def _attach_depth_stats(
    detections: list[Detection],
    depth_mm: np.ndarray,
    extractor: DepthRoiStats,
) -> list[Detection]:
    for detection in detections:
        detection.depth_stats = extractor.extract(depth_mm, detection.bbox_xyxy)
    return detections


def _crop_resize(color_bgr: np.ndarray, depth_mm: np.ndarray, roi: RoiTransform) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = roi.crop_xyxy
    color_crop = color_bgr[y0:y1, x0:x1]
    depth_crop = depth_mm[y0:y1, x0:x1]
    color_roi = cv2.resize(color_crop, (roi.roi_size, roi.roi_size), interpolation=cv2.INTER_LINEAR)
    depth_roi = cv2.resize(depth_crop, (roi.roi_size, roi.roi_size), interpolation=cv2.INTER_NEAREST)
    return color_roi, depth_roi.astype(np.float32, copy=False)


def _mask_to_image(mask_256: np.ndarray, roi: RoiTransform, height: int, width: int) -> np.ndarray:
    x0, y0, x1, y1 = roi.crop_xyxy
    resized = cv2.resize(
        mask_256.astype(np.uint8),
        (roi.crop_width, roi.crop_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    image_mask = np.zeros((height, width), dtype=bool)
    image_mask[y0:y1, x0:x1] = resized
    return image_mask


def _debug_snapshot(
    frame: DeployFrame,
    detections: list[Detection],
    target: GraspTarget,
    candidates: list[_PipelineCandidate],
) -> DebugSnapshot:
    return DebugSnapshot(
        frame_id=frame.frame_id,
        timestamp_ms=frame.timestamp_ms,
        detections=list(detections),
        masks=[candidate.image_mask for candidate in candidates],
        target=target,
    )


def _detection_record(detection: Detection) -> dict[str, Any]:
    return {
        "bbox_xyxy": list(detection.bbox_xyxy),
        "class_id": detection.class_id,
        "confidence": detection.confidence,
        "weighted_confidence": detection.weighted_confidence,
        "label": detection.label,
        "track_id": detection.track_id,
        "depth_stats": asdict(detection.depth_stats),
    }


def _candidate_image_coordinates(candidate: _PipelineCandidate, frame: DeployFrame) -> tuple[float, float, float]:
    u_img, v_img = candidate.roi_transform.roi_to_image(candidate.patch.u_px, candidate.patch.v_px)
    z_mm = candidate.patch.z_m / float(frame.intrinsics.depth_scale)
    return float(u_img), float(v_img), float(z_mm)


def _rejection_record(
    track: Track,
    stage: str,
    reason: str,
    *,
    metric: str | None = None,
    value: float | None = None,
    threshold: float | None = None,
) -> dict[str, Any]:
    return {
        "track_id": track.track_id,
        "bbox_xyxy": list(track.bbox_xyxy),
        "stage": stage,
        "reason": reason,
        "metric": metric,
        "value": value,
        "threshold": threshold,
    }


def _elapsed_ms(started: float) -> float:
    return float((perf_counter() - started) * 1000.0)
