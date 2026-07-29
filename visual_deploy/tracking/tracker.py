from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from visual_deploy.types import BBoxXYXY, DepthStats, Detection, Track


@dataclass
class ConfidenceGate:
    high_threshold: float = 0.7
    low_threshold: float = 0.4

    def apply(self, detections: list[Detection]) -> list[Detection]:
        kept: list[Detection] = []
        span = max(self.high_threshold - self.low_threshold, 1e-6)

        for detection in detections:
            confidence = float(detection.confidence)
            if confidence < self.low_threshold:
                continue

            if confidence >= self.high_threshold:
                detection.weighted_confidence = confidence
            else:
                weight = (confidence - self.low_threshold) / span
                detection.weighted_confidence = confidence * weight

            kept.append(detection)

        return kept


@dataclass
class DepthRoiStats:
    min_depth_mm: float = 100.0
    max_depth_mm: float = 5000.0

    def extract(self, depth_mm: np.ndarray | None, bbox: BBoxXYXY) -> DepthStats:
        if depth_mm is None:
            return DepthStats(state="invalid")

        if depth_mm.ndim < 2 or depth_mm.ndim > 3:
            return DepthStats(state="invalid")

        if depth_mm.ndim == 3:
            if depth_mm.shape[2] < 1:
                return DepthStats(state="invalid")
            depth_view = depth_mm[:, :, 0]
        else:
            depth_view = depth_mm

        height, width = depth_view.shape[:2]
        x1, y1, x2, y2 = (int(round(value)) for value in bbox)
        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return DepthStats(state="invalid")

        roi = depth_view[y1:y2, x1:x2].astype(np.float32, copy=False)
        values = roi.reshape(-1)
        valid_mask = np.isfinite(values)
        valid_mask &= values > 0
        valid_mask &= values >= self.min_depth_mm
        valid_mask &= values <= self.max_depth_mm

        valid_values = values[valid_mask]
        if valid_values.size == 0:
            return DepthStats(state="invalid")

        valid_ratio = float(valid_values.size / values.size)
        return DepthStats(
            median_mm=float(np.median(valid_values)),
            mean_mm=float(np.mean(valid_values)),
            std_mm=float(np.std(valid_values)),
            valid_ratio=valid_ratio,
            valid_count=int(valid_values.size),
            state="valid" if valid_ratio >= 0.5 else "partial",
        )


@dataclass
class _TrackState:
    track_id: int
    bbox_xyxy: BBoxXYXY
    class_id: int
    label: str
    confidence: float
    hits: int = 1
    lost: int = 0


def _iou(box_a: BBoxXYXY, box_b: BBoxXYXY) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area / union)


class SimpleIoUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 30, min_hits: int = 2) -> None:
        self._iou_threshold = iou_threshold
        self._max_lost = max_lost
        self._min_hits = min_hits
        self._next_track_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        outputs: list[Track] = []
        remaining = set(range(len(detections)))
        matched_track_ids: set[int] = set()

        for track_id in list(self._tracks.keys()):
            track = self._tracks[track_id]
            best_idx: int | None = None
            best_score = -1.0

            for det_idx in remaining:
                detection = detections[det_idx]
                if detection.class_id != track.class_id:
                    continue

                score = _iou(track.bbox_xyxy, detection.bbox_xyxy)
                if score > best_score:
                    best_idx = det_idx
                    best_score = score

            if best_idx is None or best_score < self._iou_threshold:
                continue

            detection = detections[best_idx]
            track.bbox_xyxy = detection.bbox_xyxy
            track.label = detection.label
            track.confidence = _effective_confidence(detection)
            track.hits += 1
            track.lost = 0

            detection.track_id = track_id
            remaining.remove(best_idx)
            matched_track_ids.add(track_id)
            outputs.append(self._to_output(track, detection))

        for det_idx in sorted(remaining):
            detection = detections[det_idx]
            track_id = self._next_track_id
            self._next_track_id += 1
            track = _TrackState(
                track_id=track_id,
                bbox_xyxy=detection.bbox_xyxy,
                class_id=detection.class_id,
                label=detection.label,
                confidence=_effective_confidence(detection),
            )

            self._tracks[track_id] = track
            matched_track_ids.add(track_id)
            detection.track_id = track_id
            outputs.append(self._to_output(track, detection))

        for track_id, track in list(self._tracks.items()):
            if track_id in matched_track_ids:
                continue
            track.lost += 1
            if track.lost > self._max_lost:
                del self._tracks[track_id]

        return outputs

    def _to_output(self, track: _TrackState, detection: Detection) -> Track:
        return Track(
            track_id=track.track_id,
            bbox_xyxy=track.bbox_xyxy,
            class_id=track.class_id,
            confidence=track.confidence,
            label=track.label,
            state="confirmed" if track.hits >= self._min_hits else "tentative",
            hits=track.hits,
            lost=track.lost,
            depth_stats=detection.depth_stats,
            weighted_confidence=detection.weighted_confidence,
        )


class DualThresholdIoUTracker:
    """High-confidence detections create tracks; low-confidence detections only continue them."""

    def __init__(
        self,
        high_threshold: float = 0.5,
        low_threshold: float = 0.1,
        iou_threshold: float = 0.3,
        max_lost: int = 30,
        min_hits: int = 1,
    ) -> None:
        self.high_threshold = _unit_threshold(high_threshold, "high_threshold")
        self.low_threshold = _unit_threshold(low_threshold, "low_threshold")
        if self.low_threshold > self.high_threshold:
            raise ValueError("low_threshold must be less than or equal to high_threshold")
        self.iou_threshold = _unit_threshold(iou_threshold, "iou_threshold")
        if isinstance(max_lost, bool) or int(max_lost) != max_lost or max_lost < 0:
            raise ValueError("max_lost must be a non-negative integer")
        if isinstance(min_hits, bool) or int(min_hits) != min_hits or min_hits < 1:
            raise ValueError("min_hits must be a positive integer")
        self.max_lost = int(max_lost)
        self.min_hits = int(min_hits)
        self._next_track_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        for detection in detections:
            _unit_threshold(detection.confidence, "detection.confidence")
            detection.weighted_confidence = float(detection.confidence)
        high_indices = {i for i, det in enumerate(detections) if det.confidence >= self.high_threshold}
        low_indices = {
            i for i, det in enumerate(detections) if self.low_threshold <= det.confidence < self.high_threshold
        }
        existing_track_ids = set(self._tracks)
        unmatched_track_ids = set(existing_track_ids)
        matched_track_ids: set[int] = set()
        outputs: list[Track] = []

        for detection_indices in (high_indices, low_indices):
            for track_id, detection_index in self._match(unmatched_track_ids, detection_indices, detections):
                track = self._tracks[track_id]
                detection = detections[detection_index]
                track.bbox_xyxy = detection.bbox_xyxy
                track.class_id = detection.class_id
                track.confidence = float(detection.confidence)
                track.label = detection.label
                track.hits += 1
                track.lost = 0
                detection.track_id = track_id
                unmatched_track_ids.remove(track_id)
                detection_indices.remove(detection_index)
                matched_track_ids.add(track_id)
                outputs.append(self._to_output(track, detection))

        for detection_index in sorted(high_indices):
            detection = detections[detection_index]
            track = _TrackState(
                self._next_track_id,
                detection.bbox_xyxy,
                detection.class_id,
                detection.label,
                float(detection.confidence),
            )
            self._tracks[track.track_id] = track
            self._next_track_id += 1
            detection.track_id = track.track_id
            matched_track_ids.add(track.track_id)
            outputs.append(self._to_output(track, detection))

        for track_id in existing_track_ids - matched_track_ids:
            track = self._tracks[track_id]
            track.lost += 1
            if track.lost > self.max_lost:
                del self._tracks[track_id]
        return outputs

    def _match(
        self,
        track_ids: set[int],
        detection_indices: set[int],
        detections: list[Detection],
    ) -> list[tuple[int, int]]:
        remaining = set(detection_indices)
        matches: list[tuple[int, int]] = []
        for track_id in sorted(track_ids):
            track = self._tracks[track_id]
            candidates = [
                (index, _iou(track.bbox_xyxy, detections[index].bbox_xyxy))
                for index in remaining
                if detections[index].class_id == track.class_id
            ]
            if not candidates:
                continue
            detection_index, score = max(candidates, key=lambda item: (item[1], -item[0]))
            if score >= self.iou_threshold:
                matches.append((track_id, detection_index))
                remaining.remove(detection_index)
        return matches

    def get_coasting_track(self, track_id: int, max_coast_frames: int) -> Track | None:
        if isinstance(max_coast_frames, bool) or not isinstance(max_coast_frames, int) or max_coast_frames < 0:
            raise ValueError("max_coast_frames must be a non-negative integer")
        track = self._tracks.get(int(track_id))
        if (
            track is None
            or track.hits < self.min_hits
            or track.lost <= 0
            or track.lost > min(max_coast_frames, self.max_lost)
        ):
            return None
        return Track(
            track_id=track.track_id,
            bbox_xyxy=track.bbox_xyxy,
            class_id=track.class_id,
            confidence=track.confidence,
            label=track.label,
            state="coasting",
            hits=track.hits,
            lost=track.lost,
            weighted_confidence=track.confidence,
        )

    def _to_output(self, track: _TrackState, detection: Detection) -> Track:
        return Track(
            track_id=track.track_id,
            bbox_xyxy=track.bbox_xyxy,
            class_id=track.class_id,
            confidence=track.confidence,
            label=track.label,
            state="confirmed" if track.hits >= self.min_hits else "tentative",
            hits=track.hits,
            lost=track.lost,
            depth_stats=detection.depth_stats,
            weighted_confidence=detection.weighted_confidence,
        )


def _effective_confidence(detection: Detection) -> float:
    weighted_confidence = detection.weighted_confidence
    if weighted_confidence is not None:
        return float(weighted_confidence)
    return float(detection.confidence)


def _unit_threshold(value: float, name: str) -> float:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0 or numeric > 1.0:
        raise ValueError(f"{name} must be finite and in [0.0, 1.0]")
    return numeric
