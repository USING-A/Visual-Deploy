# RGB-D Grasp Deploy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `Visual-Deploy` into a lightweight, self-contained RealSense RGB-D deployment project that detects apples, segments confirmed tracks, selects suction grasp patches, records intermediate boxes/masks, and outputs one camera-frame grasp target.

**Architecture:** Keep `Visual-Deploy` self-contained at runtime. Port only minimal runtime algorithms from the research repositories, load exported YOLO and reparameterized GCNet inference weights, and validate geometry/depth contracts with focused tests before realtime integration.

**Tech Stack:** Python 3.10+, PyTorch, OpenCV, NumPy, PyYAML, pytest, optional `pyrealsense2`, optional Ultralytics or exported YOLO runtime.

---

## Reference Spec

Read before implementation:

- `docs/superpowers/specs/2026-07-02-rgbd-grasp-deploy-design.md`

Critical constraints:

- Final `Visual-Deploy` runtime must not import `D:\Github Code\yolov10-improved`, `D:\Github Code\yolo-multi-frame-postprocess`, or `D:\Github Code\RGBD-GCNet`.
- Development may use those repositories for export and verification.
- Keep implementation commits small and focused.
- Before every commit, run `git status --short` and focused tests for touched files.

## File Structure

Create these runtime modules:

```text
visual_deploy/
  __init__.py
  config.py
  types.py
  camera/
    __init__.py
    realsense_source.py
  detection/
    __init__.py
    yolo_detector.py
  tracking/
    __init__.py
    tracker.py
    depth_fusion.py
  segmentation/
    __init__.py
    gcnet_segmentor.py
  geometry/
    __init__.py
    roi.py
    backproject.py
    grasp_patch.py
    pose.py
  ranking/
    __init__.py
    target_ranker.py
  recording/
    __init__.py
    recorder.py
  pipeline/
    __init__.py
    offline_pipeline.py
    realtime_pipeline.py
scripts/
  export_gcnet_l03_inference.py
  run_offline_smoke.py
  run_realtime.py
  verify_mapping.py
configs/
  deploy.yaml
tests/
```

Use this responsibility split:

- `types.py`: dataclasses shared across modules.
- `config.py`: YAML loading and path resolution.
- `geometry/`: all ROI, intrinsics, backprojection, patch, and camera-layer pose logic.
- `tracking/`: confidence gate, IoU tracker, per-track depth fusion.
- `segmentation/`: load exported GCNet inference `.pt`.
- `detection/`: YOLO loader abstraction.
- `recording/`: JSONL and artifact writing.
- `pipeline/`: orchestration only.

---

### Task 1: Scaffold Package, Config, And Shared Types

**Model routing:** `gpt-5.4`, medium effort.

**Files:**
- Create: `pyproject.toml`
- Create: `configs/deploy.yaml`
- Create: `visual_deploy/__init__.py`
- Create: `visual_deploy/config.py`
- Create: `visual_deploy/types.py`
- Create: package `__init__.py` files under all module folders
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing config and type tests**

Create `tests/test_config.py`:

```python
from pathlib import Path

from visual_deploy.config import load_config, resolve_path
from visual_deploy.types import CameraIntrinsics, Detection


def test_load_default_config_has_runtime_sections():
    cfg = load_config(Path("configs/deploy.yaml"))
    assert cfg["roi"]["size"] == 256
    assert cfg["roi"]["pad_ratio"] == 0.20
    assert cfg["camera"]["align_to_color"] is True
    assert cfg["segmentation"]["depth_input_unit"] == "mm"


def test_resolve_path_is_relative_to_project_root(tmp_path):
    cfg_path = tmp_path / "configs" / "deploy.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text("weights: weights/model.pt\n", encoding="utf-8")
    assert resolve_path(cfg_path, "weights/model.pt") == tmp_path / "weights" / "model.pt"


def test_shared_types_are_plain_dataclasses():
    intr = CameraIntrinsics(fx=600.0, fy=601.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    det = Detection(bbox_xyxy=(1.0, 2.0, 3.0, 4.0), class_id=0, confidence=0.9, label="apple")
    assert intr.fx == 600.0
    assert det.weighted_confidence == 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_config.py -v
```

Expected: FAIL because `visual_deploy.config` and dataclasses do not exist.

- [ ] **Step 3: Add packaging metadata**

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "visual-deploy"
version = "0.1.0"
description = "Self-contained RGB-D visual grasp deployment pipeline"
requires-python = ">=3.10"
dependencies = [
  "numpy>=1.24",
  "opencv-python>=4.8",
  "pyyaml>=6.0",
  "torch>=2.0",
]

[project.optional-dependencies]
dev = ["pytest>=7.4"]
realsense = ["pyrealsense2>=2.55.1"]
yolo = ["ultralytics>=8.2.0"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 4: Add default config**

Create `configs/deploy.yaml`:

```yaml
camera:
  width: 640
  height: 480
  fps: 30
  align_to_color: true
  spatial_filter: true
  temporal_filter: false
  hole_filling_filter: false

detection:
  weights: weights/yolo_detect.pt
  conf_threshold: 0.25
  device: cuda:0

tracking:
  high_conf_threshold: 0.7
  low_conf_threshold: 0.4
  iou_threshold: 0.3
  min_hits: 2
  max_lost: 30

roi:
  size: 256
  pad_ratio: 0.20

segmentation:
  weights: weights/rgbd_gcnet_l03_robustft_inference.pt
  threshold: 0.5
  device: cuda:0
  input_color_order: bgr
  depth_input_unit: mm

depth_fusion:
  window_size: 5
  method: median
  min_depth_mm: 100
  max_depth_mm: 5000
  min_valid_ratio: 0.3

grasp:
  patch_radius_px: 8
  stride_px: 4
  min_component_area_px: 64
  min_valid_depth_ratio: 0.7
  min_valid_depth_count: 20
  max_plane_rmse_m: 0.006
  max_depth_mad_m: 0.012
  min_score: 0.0

ranking:
  weights:
    grasp_score: 0.45
    depth_valid_score: 0.20
    track_confidence: 0.15
    track_stability: 0.10
    mask_quality_score: 0.10

recording:
  output_root: runs
  save_frames: false
  save_depth_preview: true
  save_masks: true
  save_overlays: true
```

- [ ] **Step 5: Add config loader**

Create `visual_deploy/config.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}")
    return data


def resolve_path(config_path: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    config_path = Path(config_path)
    root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    return root / path
```

- [ ] **Step 6: Add shared dataclasses**

Create `visual_deploy/types.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    ppx: float
    ppy: float
    depth_scale: float = 0.001


@dataclass
class DeployFrame:
    frame_id: int
    timestamp_ms: float
    color_bgr: np.ndarray
    depth_mm: np.ndarray
    intrinsics: CameraIntrinsics
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class DepthStats:
    median_mm: float | None = None
    mean_mm: float | None = None
    std_mm: float | None = None
    valid_ratio: float = 0.0
    valid_count: int = 0
    state: str = "invalid"


@dataclass
class Detection:
    bbox_xyxy: BBoxXYXY
    class_id: int
    confidence: float
    label: str = ""
    track_id: int | None = None
    weighted_confidence: float | None = None
    depth_stats: DepthStats = field(default_factory=DepthStats)

    def __post_init__(self) -> None:
        if self.weighted_confidence is None:
            self.weighted_confidence = float(self.confidence)


@dataclass
class Track:
    track_id: int
    bbox_xyxy: BBoxXYXY
    class_id: int
    confidence: float
    label: str
    state: str
    hits: int
    lost: int
    depth_stats: DepthStats = field(default_factory=DepthStats)
    weighted_confidence: float | None = None


@dataclass
class GraspTarget:
    valid: bool
    frame_id: int
    reason: str | None = None
    track_id: int | None = None
    u_px: float | None = None
    v_px: float | None = None
    z_mm: float | None = None
    xyz_camera_m: tuple[float, float, float] | None = None
    normal_xyz: tuple[float, float, float] | None = None
    approach_axis: tuple[float, float, float] | None = None
    target_score: float | None = None
```

Create empty package files:

```text
visual_deploy/__init__.py
visual_deploy/camera/__init__.py
visual_deploy/detection/__init__.py
visual_deploy/tracking/__init__.py
visual_deploy/segmentation/__init__.py
visual_deploy/geometry/__init__.py
visual_deploy/ranking/__init__.py
visual_deploy/recording/__init__.py
visual_deploy/pipeline/__init__.py
```

- [ ] **Step 7: Run test to verify it passes**

Run:

```powershell
python -m pytest tests/test_config.py -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```powershell
git status --short
git add pyproject.toml configs/deploy.yaml visual_deploy tests/test_config.py
git commit -m "feat: scaffold deploy package"
```

Expected: commit includes only scaffold, config, shared types, and tests.

---

### Task 2: Geometry, ROI Mapping, And Backprojection

**Model routing:** `gpt-5.5`, high effort. Geometry correctness is deployment-critical.

**Files:**
- Create: `visual_deploy/geometry/roi.py`
- Create: `visual_deploy/geometry/backproject.py`
- Test: `tests/test_geometry_roi.py`

- [ ] **Step 1: Write failing geometry tests**

Create `tests/test_geometry_roi.py`:

```python
import numpy as np
import pytest

from visual_deploy.geometry.backproject import backproject_pixel
from visual_deploy.geometry.roi import RoiTransform, square_pad_bbox
from visual_deploy.types import CameraIntrinsics


def test_square_pad_bbox_uses_larger_side_with_margin():
    assert square_pad_bbox((10, 20, 50, 40), 0.2, 200, 100) == (6, 6, 54, 54)


def test_roi_round_trip_center_bbox():
    roi = RoiTransform.from_bbox((100, 50, 300, 250), 0.2, 640, 480, 256)
    u_roi, v_roi = roi.image_to_roi(200.0, 150.0)
    u_img, v_img = roi.roi_to_image(u_roi, v_roi)
    assert u_img == pytest.approx(200.0)
    assert v_img == pytest.approx(150.0)


def test_edge_clipped_bbox_keeps_independent_scales():
    roi = RoiTransform.from_bbox((0, 0, 120, 40), 0.2, 160, 80, 256)
    assert roi.crop_xyxy[0] == 0
    assert roi.crop_xyxy[1] == 0
    assert roi.scale_x != roi.scale_y


def test_roi_intrinsics_match_crop_then_resize():
    intr = CameraIntrinsics(fx=600.0, fy=610.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    roi = RoiTransform(crop_xyxy=(100, 50, 300, 250), roi_size=256)
    cam_roi = roi.adjust_intrinsics(intr)
    assert cam_roi.fx == pytest.approx(600.0 * roi.scale_x)
    assert cam_roi.fy == pytest.approx(610.0 * roi.scale_y)
    assert cam_roi.ppx == pytest.approx((320.0 - 100.0) * roi.scale_x)
    assert cam_roi.ppy == pytest.approx((240.0 - 50.0) * roi.scale_y)
    assert cam_roi.depth_scale == pytest.approx(0.001)


def test_roi_and_full_backprojection_are_consistent():
    intr = CameraIntrinsics(fx=600.0, fy=610.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    roi = RoiTransform(crop_xyxy=(100, 50, 300, 250), roi_size=256)
    cam_roi = roi.adjust_intrinsics(intr)
    u_img, v_img, depth_mm = 210.0, 170.0, 500.0
    u_roi, v_roi = roi.image_to_roi(u_img, v_img)
    xyz_full = backproject_pixel(u_img, v_img, depth_mm, intr)
    xyz_roi = backproject_pixel(u_roi, v_roi, depth_mm, cam_roi)
    np.testing.assert_allclose(xyz_roi, xyz_full, atol=1e-6)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_geometry_roi.py -v
```

Expected: FAIL because geometry modules do not exist.

- [ ] **Step 3: Implement ROI transform**

Create `visual_deploy/geometry/roi.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from visual_deploy.types import BBoxXYXY, CameraIntrinsics


def square_pad_bbox(bbox: BBoxXYXY, pad_ratio: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    side = max(x1 - x0, y1 - y0) * (1.0 + float(pad_ratio))
    half = side / 2.0
    nx0 = max(0, int(round(cx - half)))
    ny0 = max(0, int(round(cy - half)))
    nx1 = min(int(img_w), int(round(cx + half)))
    ny1 = min(int(img_h), int(round(cy + half)))
    return nx0, ny0, nx1, ny1


@dataclass(frozen=True)
class RoiTransform:
    crop_xyxy: tuple[int, int, int, int]
    roi_size: int = 256

    @classmethod
    def from_bbox(
        cls,
        bbox: BBoxXYXY,
        pad_ratio: float,
        img_w: int,
        img_h: int,
        roi_size: int = 256,
    ) -> "RoiTransform":
        return cls(square_pad_bbox(bbox, pad_ratio, img_w, img_h), roi_size=roi_size)

    @property
    def crop_width(self) -> int:
        return int(self.crop_xyxy[2] - self.crop_xyxy[0])

    @property
    def crop_height(self) -> int:
        return int(self.crop_xyxy[3] - self.crop_xyxy[1])

    @property
    def scale_x(self) -> float:
        return float(self.roi_size) / float(self.crop_width)

    @property
    def scale_y(self) -> float:
        return float(self.roi_size) / float(self.crop_height)

    def image_to_roi(self, u_img: float, v_img: float) -> tuple[float, float]:
        x0, y0, _, _ = self.crop_xyxy
        return (float(u_img) - x0) * self.scale_x, (float(v_img) - y0) * self.scale_y

    def roi_to_image(self, u_roi: float, v_roi: float) -> tuple[float, float]:
        x0, y0, _, _ = self.crop_xyxy
        return float(u_roi) / self.scale_x + x0, float(v_roi) / self.scale_y + y0

    def adjust_intrinsics(self, intr: CameraIntrinsics) -> CameraIntrinsics:
        x0, y0, _, _ = self.crop_xyxy
        return CameraIntrinsics(
            fx=float(intr.fx) * self.scale_x,
            fy=float(intr.fy) * self.scale_y,
            ppx=(float(intr.ppx) - x0) * self.scale_x,
            ppy=(float(intr.ppy) - y0) * self.scale_y,
            depth_scale=float(intr.depth_scale),
        )
```

- [ ] **Step 4: Implement backprojection**

Create `visual_deploy/geometry/backproject.py`:

```python
from __future__ import annotations

import numpy as np

from visual_deploy.types import CameraIntrinsics


def backproject_pixel(u: float, v: float, depth_raw: float, intr: CameraIntrinsics) -> np.ndarray:
    z = float(depth_raw) * float(intr.depth_scale)
    x = (float(u) - float(intr.ppx)) * z / float(intr.fx)
    y = (float(v) - float(intr.ppy)) * z / float(intr.fy)
    return np.asarray([x, y, z], dtype=np.float32)
```

- [ ] **Step 5: Run geometry tests**

Run:

```powershell
python -m pytest tests/test_geometry_roi.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git status --short
git add visual_deploy/geometry/roi.py visual_deploy/geometry/backproject.py tests/test_geometry_roi.py
git commit -m "feat: add ROI geometry mapping"
```

---

### Task 3: Patch Grasp And Camera-Layer Pose

**Model routing:** `gpt-5.5`, high effort. This ports the grasp selector contract.

**Files:**
- Create: `visual_deploy/geometry/grasp_patch.py`
- Create: `visual_deploy/geometry/pose.py`
- Test: `tests/test_grasp_patch.py`

- [ ] **Step 1: Write failing grasp tests**

Create `tests/test_grasp_patch.py`:

```python
import numpy as np
import pytest

from visual_deploy.geometry.grasp_patch import select_grasp_patch
from visual_deploy.geometry.pose import approach_from_normal
from visual_deploy.types import CameraIntrinsics


def test_select_grasp_patch_prefers_flat_interior_depth_support_mm():
    mask = np.zeros((64, 64), dtype=bool)
    yy, xx = np.ogrid[:64, :64]
    mask[(yy - 32) ** 2 + (xx - 32) ** 2 <= 20 ** 2] = True
    depth = np.full((64, 64), np.nan, dtype=np.float32)
    depth[mask] = 800.0
    depth[mask & (xx > 42)] = 900.0
    intr = CameraIntrinsics(fx=100.0, fy=100.0, ppx=32.0, ppy=32.0, depth_scale=0.001)

    out = select_grasp_patch(mask, depth, intr, patch_radius_px=5, stride_px=4)

    assert out is not None
    assert abs(out.u_px - 32.0) <= 4.0
    assert abs(out.v_px - 32.0) <= 4.0
    assert out.z_m == pytest.approx(0.8, abs=0.05)
    assert out.valid_depth_ratio >= 0.95
    assert out.normal_xyz[2] <= 0.0


def test_select_grasp_patch_rejects_unreliable_depth_support():
    mask = np.zeros((48, 48), dtype=bool)
    mask[12:36, 12:36] = True
    depth = np.full((48, 48), np.nan, dtype=np.float32)
    depth[20:24, 20:24] = 700.0
    intr = CameraIntrinsics(fx=100.0, fy=100.0, ppx=24.0, ppy=24.0, depth_scale=0.001)

    out = select_grasp_patch(mask, depth, intr, patch_radius_px=5, stride_px=4, min_valid_depth_ratio=0.6)

    assert out is None


def test_approach_axis_is_negative_normal_for_suction():
    normal = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    approach = approach_from_normal(normal)
    np.testing.assert_allclose(approach, np.array([-0.0, -0.0, 1.0], dtype=np.float32))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_grasp_patch.py -v
```

Expected: FAIL because grasp modules do not exist.

- [ ] **Step 3: Port patch selector**

Implement `visual_deploy/geometry/grasp_patch.py` by porting the deterministic algorithm from `RGBD-GCNet/apple_grasp/patch_selector.py` with these local changes:

```python
@dataclass(frozen=True)
class GraspPatch:
    u_px: float
    v_px: float
    z_m: float
    normal_xyz: np.ndarray
    score: float
    valid_depth_ratio: float
    valid_depth_count: int
    plane_rmse_m: float
    depth_mad_m: float
    boundary_distance_px: float
    component_area_px: int
```

The public function signature must be:

```python
def select_grasp_patch(
    mask: np.ndarray,
    depth_mm: np.ndarray,
    intr: CameraIntrinsics,
    *,
    patch_radius_px: int = 8,
    stride_px: int = 4,
    min_component_area_px: int = 64,
    min_valid_depth_ratio: float = 0.7,
    min_valid_depth_count: int = 20,
    min_boundary_distance_px: float | None = None,
    min_score: float = 0.0,
    max_plane_rmse_m: float = 0.006,
    max_depth_mad_m: float = 0.012,
) -> GraspPatch | None:
```

Inside it, convert depth to meters only through `intr.depth_scale` when backprojecting and calculating `z_m`.

- [ ] **Step 4: Implement pose helper**

Create `visual_deploy/geometry/pose.py`:

```python
from __future__ import annotations

import numpy as np


def approach_from_normal(normal_xyz: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal_xyz, dtype=np.float32)
    norm = float(np.linalg.norm(normal))
    if norm < 1e-9:
        raise ValueError("normal vector has near-zero length")
    return (-normal / norm).astype(np.float32)
```

- [ ] **Step 5: Run grasp tests**

Run:

```powershell
python -m pytest tests/test_grasp_patch.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```powershell
git status --short
git add visual_deploy/geometry/grasp_patch.py visual_deploy/geometry/pose.py tests/test_grasp_patch.py
git commit -m "feat: add suction patch grasp geometry"
```

---

### Task 4: Detection Gating, IoU Tracking, And Depth Stats

**Model routing:** `gpt-5.4`, medium effort.

**Files:**
- Create: `visual_deploy/tracking/tracker.py`
- Test: `tests/test_tracking.py`

- [ ] **Step 1: Write failing tracking tests**

Create `tests/test_tracking.py`:

```python
import numpy as np

from visual_deploy.tracking.tracker import ConfidenceGate, DepthRoiStats, SimpleIoUTracker
from visual_deploy.types import Detection


def test_confidence_gate_drops_low_and_weights_mid_confidence():
    gate = ConfidenceGate(high_threshold=0.7, low_threshold=0.4)
    kept = gate.apply([
        Detection((0, 0, 10, 10), 0, 0.3),
        Detection((0, 0, 10, 10), 0, 0.55),
        Detection((0, 0, 10, 10), 0, 0.8),
    ])
    assert len(kept) == 2
    assert kept[0].weighted_confidence < kept[0].confidence
    assert kept[1].weighted_confidence == kept[1].confidence


def test_iou_tracker_confirms_after_min_hits():
    tracker = SimpleIoUTracker(iou_threshold=0.3, max_lost=3, min_hits=2)
    first = tracker.update([Detection((10, 10, 50, 50), 0, 0.9)])
    second = tracker.update([Detection((12, 12, 52, 52), 0, 0.8)])
    assert first[0].state == "tentative"
    assert second[0].state == "confirmed"
    assert second[0].track_id == first[0].track_id


def test_depth_roi_stats_uses_mm_thresholds():
    depth = np.array([[0, 200, 300], [6000, 400, 500]], dtype=np.float32)
    stats = DepthRoiStats(min_depth_mm=100, max_depth_mm=5000).extract(depth, (0, 0, 3, 2))
    assert stats.valid_count == 4
    assert stats.median_mm == 350.0
    assert stats.state == "valid"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_tracking.py -v
```

Expected: FAIL because tracker module does not exist.

- [ ] **Step 3: Implement tracker and depth stats**

Create `visual_deploy/tracking/tracker.py` with:

```python
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
        for det in detections:
            if det.confidence < self.low_threshold:
                continue
            if det.confidence >= self.high_threshold:
                det.weighted_confidence = det.confidence
            else:
                span = max(self.high_threshold - self.low_threshold, 1e-6)
                det.weighted_confidence = det.confidence * ((det.confidence - self.low_threshold) / span)
            kept.append(det)
        return kept


@dataclass
class DepthRoiStats:
    min_depth_mm: float = 100.0
    max_depth_mm: float = 5000.0

    def extract(self, depth_mm: np.ndarray | None, bbox: BBoxXYXY) -> DepthStats:
        if depth_mm is None:
            return DepthStats(state="invalid")
        view = depth_mm[:, :, 0] if depth_mm.ndim > 2 else depth_mm
        h, w = view.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))
        if x2 <= x1 or y2 <= y1:
            return DepthStats(state="invalid")
        roi = view[y1:y2, x1:x2].astype(np.float32)
        valid = np.isfinite(roi) & (roi > 0) & (roi >= self.min_depth_mm) & (roi <= self.max_depth_mm)
        values = roi[valid]
        if values.size == 0:
            return DepthStats(state="invalid")
        ratio = float(values.size / max(roi.size, 1))
        return DepthStats(
            median_mm=float(np.median(values)),
            mean_mm=float(np.mean(values)),
            std_mm=float(np.std(values)),
            valid_ratio=ratio,
            valid_count=int(values.size),
            state="valid" if ratio >= 0.5 else "partial",
        )
```

Also implement `_iou`, internal `_TrackState`, and `SimpleIoUTracker.update()` with the same behavior as the current `yolo-multi-frame-postprocess` tracker, returning `Track` dataclasses.

- [ ] **Step 4: Run tracking tests**

Run:

```powershell
python -m pytest tests/test_tracking.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git status --short
git add visual_deploy/tracking/tracker.py tests/test_tracking.py
git commit -m "feat: add detection tracking primitives"
```

---

### Task 5: Per-Track Depth Fusion

**Model routing:** `gpt-5.4`, medium effort.

**Files:**
- Create: `visual_deploy/tracking/depth_fusion.py`
- Test: `tests/test_depth_fusion.py`

- [ ] **Step 1: Write failing depth fusion tests**

Create `tests/test_depth_fusion.py`:

```python
import numpy as np

from visual_deploy.geometry.roi import RoiTransform
from visual_deploy.tracking.depth_fusion import DepthFusionBuffer


def test_depth_fusion_returns_current_frame_until_enough_history():
    buffer = DepthFusionBuffer(window_size=3, min_depth_mm=100, max_depth_mm=5000, min_valid_ratio=0.3)
    depth = np.full((4, 4), 500.0, dtype=np.float32)
    fused = buffer.update(track_id=1, frame_id=1, timestamp_ms=0.0, roi_transform=RoiTransform((0, 0, 4, 4), 4), depth_roi_mm=depth)
    assert fused.source_frame_count == 1
    np.testing.assert_allclose(fused.depth_roi_mm, depth)


def test_depth_fusion_uses_per_pixel_median():
    buffer = DepthFusionBuffer(window_size=3, min_depth_mm=100, max_depth_mm=5000, min_valid_ratio=0.3)
    roi = RoiTransform((0, 0, 2, 2), 2)
    for idx, value in enumerate([400.0, 500.0, 1000.0], 1):
        out = buffer.update(1, idx, float(idx), roi, np.full((2, 2), value, dtype=np.float32))
    assert out.source_frame_count == 3
    np.testing.assert_allclose(out.depth_roi_mm, np.full((2, 2), 500.0, dtype=np.float32))


def test_depth_fusion_ignores_invalid_values():
    buffer = DepthFusionBuffer(window_size=3, min_depth_mm=100, max_depth_mm=5000, min_valid_ratio=0.3)
    roi = RoiTransform((0, 0, 2, 2), 2)
    buffer.update(1, 1, 1.0, roi, np.array([[0, 500], [6000, 500]], dtype=np.float32))
    out = buffer.update(1, 2, 2.0, roi, np.array([[400, 700], [800, 900]], dtype=np.float32))
    assert out.valid_ratio >= 0.5
    assert np.isfinite(out.depth_roi_mm).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_depth_fusion.py -v
```

Expected: FAIL because depth fusion module does not exist.

- [ ] **Step 3: Implement DepthFusionBuffer**

Create `visual_deploy/tracking/depth_fusion.py`:

```python
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from visual_deploy.geometry.roi import RoiTransform


@dataclass
class FusedTrackDepth:
    track_id: int
    depth_roi_mm: np.ndarray
    valid_ratio: float
    source_frame_count: int
    age_ms: float


@dataclass
class _DepthEntry:
    frame_id: int
    timestamp_ms: float
    roi_transform: RoiTransform
    depth_roi_mm: np.ndarray


class DepthFusionBuffer:
    def __init__(
        self,
        window_size: int = 5,
        min_depth_mm: float = 100.0,
        max_depth_mm: float = 5000.0,
        min_valid_ratio: float = 0.3,
    ) -> None:
        self.window_size = int(window_size)
        self.min_depth_mm = float(min_depth_mm)
        self.max_depth_mm = float(max_depth_mm)
        self.min_valid_ratio = float(min_valid_ratio)
        self._history: dict[int, deque[_DepthEntry]] = defaultdict(lambda: deque(maxlen=self.window_size))

    def update(
        self,
        track_id: int,
        frame_id: int,
        timestamp_ms: float,
        roi_transform: RoiTransform,
        depth_roi_mm: np.ndarray,
    ) -> FusedTrackDepth:
        depth = np.asarray(depth_roi_mm, dtype=np.float32)
        self._history[int(track_id)].append(_DepthEntry(frame_id, timestamp_ms, roi_transform, depth))
        entries = list(self._history[int(track_id)])
        stack = np.stack([entry.depth_roi_mm for entry in entries], axis=0).astype(np.float32)
        valid = np.isfinite(stack) & (stack >= self.min_depth_mm) & (stack <= self.max_depth_mm)
        masked = np.where(valid, stack, np.nan)
        with np.errstate(all="ignore"):
            fused = np.nanmedian(masked, axis=0).astype(np.float32)
        current_valid = np.isfinite(depth) & (depth >= self.min_depth_mm) & (depth <= self.max_depth_mm)
        fused = np.where(np.isfinite(fused), fused, np.where(current_valid, depth, 0.0)).astype(np.float32)
        valid_ratio = float((fused > 0).sum() / max(fused.size, 1))
        age_ms = float(timestamp_ms - entries[0].timestamp_ms) if entries else 0.0
        return FusedTrackDepth(int(track_id), fused, valid_ratio, len(entries), age_ms)

    def clear(self, track_id: int | None = None) -> None:
        if track_id is None:
            self._history.clear()
        else:
            self._history.pop(int(track_id), None)
```

- [ ] **Step 4: Run depth fusion tests**

Run:

```powershell
python -m pytest tests/test_depth_fusion.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git status --short
git add visual_deploy/tracking/depth_fusion.py tests/test_depth_fusion.py
git commit -m "feat: add per-track depth fusion"
```

---

### Task 6: GCNet Reparameterized Inference Export

**Model routing:** `gpt-5.5`, high effort. This task touches model export and must be reviewed by the main agent.

**Files:**
- Create: `scripts/export_gcnet_l03_inference.py`
- Create: `tests/test_gcnet_export_contract.py`

- [ ] **Step 1: Write export contract tests for local wrapper utilities**

Create `tests/test_gcnet_export_contract.py`:

```python
import torch

from scripts.export_gcnet_l03_inference import normalize_bgr_depth


def test_normalize_bgr_depth_converts_bgr_to_rgb_and_depth_mm_to_m():
    bgr = torch.zeros(1, 3, 2, 2)
    bgr[:, 0] = 10.0
    bgr[:, 1] = 20.0
    bgr[:, 2] = 30.0
    depth_mm = torch.full((1, 1, 2, 2), 500.0)
    out = normalize_bgr_depth(bgr, depth_mm)
    assert out.shape == (1, 4, 2, 2)
    expected_r = (30.0 - 123.675) / 58.395
    expected_g = (20.0 - 116.28) / 57.12
    expected_b = (10.0 - 103.53) / 57.375
    assert out[0, 0, 0, 0].item() == torch.tensor(expected_r).item()
    assert out[0, 1, 0, 0].item() == torch.tensor(expected_g).item()
    assert out[0, 2, 0, 0].item() == torch.tensor(expected_b).item()
    assert out[0, 3, 0, 0].item() == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_gcnet_export_contract.py -v
```

Expected: FAIL because export script does not exist.

- [ ] **Step 3: Implement export script with reparameterization**

Create `scripts/export_gcnet_l03_inference.py`:

```python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

RGBD_ROOT = Path(__file__).resolve().parents[2] / "RGBD-GCNet"
if RGBD_ROOT.exists() and str(RGBD_ROOT) not in sys.path:
    sys.path.insert(0, str(RGBD_ROOT))
if os.name == "nt":
    local_appdata = RGBD_ROOT / ".tmp" / "localappdata"
    local_appdata.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WIN_PD_OVERRIDE_LOCAL_APPDATA", str(local_appdata))


def normalize_bgr_depth(bgr: torch.Tensor, depth_mm: torch.Tensor) -> torch.Tensor:
    bgr = bgr.float()
    depth_mm = depth_mm.float()
    if bgr.dim() == 3:
        bgr = bgr.unsqueeze(0)
    if depth_mm.dim() == 2:
        depth_mm = depth_mm.unsqueeze(0).unsqueeze(0)
    elif depth_mm.dim() == 3:
        depth_mm = depth_mm.unsqueeze(1)
    rgb = bgr[:, [2, 1, 0], :, :]
    mean = torch.tensor([123.675, 116.28, 103.53], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    std = torch.tensor([58.395, 57.12, 57.375], dtype=rgb.dtype, device=rgb.device).view(1, 3, 1, 1)
    rgb = (rgb - mean) / std
    depth_m = depth_mm * 0.001
    return torch.cat([rgb, depth_m], dim=1)


class RGBDGCNetInferenceWrapper(nn.Module):
    def __init__(self, model: nn.Module, threshold: float = 0.5) -> None:
        super().__init__()
        self.backbone = model.backbone
        self.decode_head = model.decode_head
        self.threshold = float(threshold)

    def forward(self, bgr: torch.Tensor, depth_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = normalize_bgr_depth(bgr, depth_mm)
        feats = self.backbone(x)
        logits = self.decode_head(feats)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = torch.nn.functional.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        prob = torch.softmax(logits, dim=1)[:, 1:2]
        mask = (prob >= self.threshold).to(dtype=torch.uint8)
        return prob, mask
```

Also implement `parse_args()` and `main()`:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner

    args = parse_args()
    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmseg"))
    if "work_dir" not in cfg:
        cfg.work_dir = str(Path(args.checkpoint).parent)
    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)
    model = runner.model.to(args.device).eval()
    with torch.no_grad():
        before = model.backbone(torch.randn(1, 4, 256, 256, device=args.device))
        model.backbone.switch_to_deploy()
        after = model.backbone(torch.randn(1, 4, 256, 256, device=args.device))
    _ = before, after
    wrapper = RGBDGCNetInferenceWrapper(model, threshold=args.threshold).to(args.device).eval()
    example_bgr = torch.zeros(1, 3, 256, 256, device=args.device)
    example_depth = torch.full((1, 1, 256, 256), 500.0, device=args.device)
    traced = torch.jit.trace(wrapper, (example_bgr, example_depth), strict=False)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    traced.save(str(out_path))
    print(f"Wrote GCNet inference model to {out_path}")


if __name__ == "__main__":
    main()
```

The main agent must later strengthen the random-input sanity check to compare same input before/after `switch_to_deploy()` and exported wrapper outputs.

- [ ] **Step 4: Run local wrapper test**

Run:

```powershell
python -m pytest tests/test_gcnet_export_contract.py -v
```

Expected: PASS.

- [ ] **Step 5: Run export manually against l03 robustFT checkpoint**

Use the actual checkpoint path after confirming it exists. Example command:

```powershell
python scripts/export_gcnet_l03_inference.py `
  --config "D:\Github Code\RGBD-GCNet\configs\fuji\seg\rgbd_gcnet_s_roi_cme_afm_boundary_l03_apple_robust_aug_train_only.py" `
  --checkpoint "D:\Github Code\RGBD-GCNet\work_dirs\apple_roi\finetune\l03_apple_robust_aug_train_only_20260626\best_mIoU_iter_3000.pth" `
  --output "weights\rgbd_gcnet_l03_robustft_inference.pt" `
  --device cpu
```

Expected: exported `.pt` appears under `weights/`. If the checkpoint path differs, locate the actual best checkpoint and record the path in the commit message or progress note.

- [ ] **Step 6: Commit script and tests, not large weights**

Run:

```powershell
git status --short
git add scripts/export_gcnet_l03_inference.py tests/test_gcnet_export_contract.py
git commit -m "feat: add GCNet inference export script"
```

Do not commit model weights unless explicitly requested.

---

### Task 7: GCNet Segmentor Loader

**Model routing:** `gpt-5.4`, medium effort.

**Files:**
- Create: `visual_deploy/segmentation/gcnet_segmentor.py`
- Test: `tests/test_gcnet_segmentor.py`

- [ ] **Step 1: Write failing segmentor loader test**

Create `tests/test_gcnet_segmentor.py`:

```python
import numpy as np
import torch

from visual_deploy.segmentation.gcnet_segmentor import GCNetSegmentor


class TinyScriptableSegmentor(torch.nn.Module):
    def forward(self, bgr, depth_mm):
        prob = torch.ones((1, 1, 256, 256), dtype=torch.float32) * 0.75
        mask = (prob >= 0.5).to(torch.uint8)
        return prob, mask


def test_gcnet_segmentor_loads_torchscript_and_returns_numpy(tmp_path):
    path = tmp_path / "tiny.pt"
    scripted = torch.jit.trace(TinyScriptableSegmentor(), (torch.zeros(1, 3, 256, 256), torch.zeros(1, 1, 256, 256)))
    scripted.save(str(path))
    seg = GCNetSegmentor(path, device="cpu")
    color = np.zeros((256, 256, 3), dtype=np.uint8)
    depth = np.ones((256, 256), dtype=np.float32) * 500.0
    out = seg.infer(color, depth)
    assert out.prob_256.shape == (256, 256)
    assert out.mask_256.dtype == bool
    assert out.mask_256.all()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_gcnet_segmentor.py -v
```

Expected: FAIL because loader does not exist.

- [ ] **Step 3: Implement segmentor loader**

Create `visual_deploy/segmentation/gcnet_segmentor.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass
class SegmentResult:
    prob_256: np.ndarray
    mask_256: np.ndarray
    mask_area: int
    largest_component_ratio: float


def _largest_component_ratio(mask: np.ndarray) -> float:
    total = int(mask.sum())
    if total <= 0:
        return 0.0
    # Keep this simple for phase 1; full component extraction is in grasp selector.
    return 1.0


class GCNetSegmentor:
    def __init__(self, weights_path: str | Path, device: str = "cpu") -> None:
        self.device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
        self.model = torch.jit.load(str(weights_path), map_location=self.device)
        self.model.eval()

    def infer(self, color_bgr_256: np.ndarray, depth_mm_256: np.ndarray) -> SegmentResult:
        color = np.asarray(color_bgr_256)
        depth = np.asarray(depth_mm_256, dtype=np.float32)
        if color.shape[:2] != (256, 256) or depth.shape != (256, 256):
            raise ValueError("GCNetSegmentor expects 256x256 color and depth ROI")
        bgr = torch.from_numpy(color.transpose(2, 0, 1)).unsqueeze(0).to(self.device).float()
        depth_t = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0).to(self.device).float()
        with torch.no_grad():
            prob_t, mask_t = self.model(bgr, depth_t)
        prob = prob_t[0, 0].detach().cpu().numpy().astype(np.float32)
        mask = mask_t[0, 0].detach().cpu().numpy().astype(bool)
        return SegmentResult(prob, mask, int(mask.sum()), _largest_component_ratio(mask))
```

- [ ] **Step 4: Run segmentor tests**

Run:

```powershell
python -m pytest tests/test_gcnet_segmentor.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git status --short
git add visual_deploy/segmentation/gcnet_segmentor.py tests/test_gcnet_segmentor.py
git commit -m "feat: add GCNet segmentor loader"
```

---

### Task 8: YOLO Detector Loader

**Model routing:** `gpt-5.4-mini` for read-only dependency check, then `gpt-5.4` for implementation.

**Files:**
- Create: `visual_deploy/detection/yolo_detector.py`
- Test: `tests/test_yolo_detector.py`

- [ ] **Step 1: Read-only subagent check**

Dispatch a `gpt-5.4-mini` low-effort read-only subagent:

```text
Scope: inspect yolov10-improved pyproject and any available exported weights.
Goal: report whether standard Ultralytics YOLO loading is likely sufficient, and list exact import/package constraints.
Do not edit files.
Return: summary with paths read and uncertainties.
```

- [ ] **Step 2: Write mockable detector tests**

Create `tests/test_yolo_detector.py`:

```python
import numpy as np

from visual_deploy.detection.yolo_detector import MockDetector


def test_mock_detector_returns_center_box():
    det = MockDetector(confidence=0.8)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    out = det.infer(image)
    assert len(out) == 1
    assert out[0].bbox_xyxy == (50.0, 25.0, 150.0, 75.0)
    assert out[0].confidence == 0.8
```

- [ ] **Step 3: Implement detector abstraction**

Create `visual_deploy/detection/yolo_detector.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

from visual_deploy.types import Detection

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


class MockDetector:
    def __init__(self, confidence: float = 0.9) -> None:
        self.confidence = float(confidence)

    def infer(self, color_bgr: np.ndarray) -> list[Detection]:
        h, w = color_bgr.shape[:2]
        return [Detection((w * 0.25, h * 0.25, w * 0.75, h * 0.75), 0, self.confidence, label="mock")]


class UltralyticsYoloDetector:
    def __init__(self, weights_path: str | Path, conf_threshold: float = 0.25, device: str | None = None) -> None:
        if YOLO is None:
            raise ImportError("ultralytics is not installed; install the yolo extra or use an exported runtime")
        weights = Path(weights_path)
        if not weights.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights}")
        self.model = YOLO(str(weights))
        self.conf_threshold = float(conf_threshold)
        self.device = device

    def infer(self, color_bgr: np.ndarray) -> list[Detection]:
        preds = self.model.predict(source=color_bgr, conf=self.conf_threshold, verbose=False, device=self.device)
        if not preds:
            return []
        result = preds[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return []
        names = getattr(result, "names", {}) or {}
        detections: list[Detection] = []
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf.item()) if box.conf is not None else 0.0
            cls_id = int(box.cls.item()) if box.cls is not None else -1
            detections.append(Detection(tuple(float(v) for v in xyxy), cls_id, conf, label=str(names.get(cls_id, cls_id))))
        return detections
```

- [ ] **Step 4: Run detector tests**

Run:

```powershell
python -m pytest tests/test_yolo_detector.py -v
```

Expected: PASS.

- [ ] **Step 5: Run clean weight smoke when weight exists**

Run only if `weights/yolo_detect.pt` exists:

```powershell
@'
import numpy as np
from visual_deploy.detection.yolo_detector import UltralyticsYoloDetector
d = UltralyticsYoloDetector("weights/yolo_detect.pt", conf_threshold=0.25, device="cpu")
print(d.infer(np.zeros((640, 640, 3), dtype=np.uint8))[:1])
'@ | python -
```

Expected: model loads and inference returns a list. If it fails due to custom model definitions, record the exact error and switch to a self-contained YOLO export task before realtime integration.

- [ ] **Step 6: Commit**

Run:

```powershell
git status --short
git add visual_deploy/detection/yolo_detector.py tests/test_yolo_detector.py
git commit -m "feat: add YOLO detector adapter"
```

---

### Task 9: Target Ranking

**Model routing:** `gpt-5.4`, medium effort.

**Files:**
- Create: `visual_deploy/ranking/target_ranker.py`
- Test: `tests/test_target_ranker.py`

- [ ] **Step 1: Write failing ranking tests**

Create `tests/test_target_ranker.py`:

```python
from visual_deploy.ranking.target_ranker import CandidateScores, TargetRanker


def test_target_ranker_chooses_highest_weighted_score():
    ranker = TargetRanker()
    low = CandidateScores(track_id=1, grasp_score=0.5, depth_valid_score=0.9, track_confidence=0.9, track_stability=1.0, mask_quality_score=1.0)
    high = CandidateScores(track_id=2, grasp_score=0.95, depth_valid_score=0.8, track_confidence=0.7, track_stability=0.8, mask_quality_score=0.7)
    best = ranker.rank([low, high])[0]
    assert best.track_id == 2
    assert best.target_score > low.target_score


def test_target_ranker_ignores_invalid_candidates():
    ranker = TargetRanker()
    invalid = CandidateScores(track_id=1, grasp_score=1.0, depth_valid_score=1.0, track_confidence=1.0, track_stability=1.0, mask_quality_score=1.0, valid=False)
    assert ranker.rank([invalid]) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_target_ranker.py -v
```

Expected: FAIL because ranker module does not exist.

- [ ] **Step 3: Implement ranker**

Create `visual_deploy/ranking/target_ranker.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CandidateScores:
    track_id: int
    grasp_score: float
    depth_valid_score: float
    track_confidence: float
    track_stability: float
    mask_quality_score: float
    valid: bool = True
    target_score: float = 0.0


class TargetRanker:
    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = weights or {
            "grasp_score": 0.45,
            "depth_valid_score": 0.20,
            "track_confidence": 0.15,
            "track_stability": 0.10,
            "mask_quality_score": 0.10,
        }

    def score(self, candidate: CandidateScores) -> float:
        return float(
            self.weights["grasp_score"] * candidate.grasp_score
            + self.weights["depth_valid_score"] * candidate.depth_valid_score
            + self.weights["track_confidence"] * candidate.track_confidence
            + self.weights["track_stability"] * candidate.track_stability
            + self.weights["mask_quality_score"] * candidate.mask_quality_score
        )

    def rank(self, candidates: list[CandidateScores]) -> list[CandidateScores]:
        valid = [candidate for candidate in candidates if candidate.valid]
        for candidate in valid:
            candidate.target_score = self.score(candidate)
        return sorted(valid, key=lambda item: item.target_score, reverse=True)
```

- [ ] **Step 4: Run ranking tests**

Run:

```powershell
python -m pytest tests/test_target_ranker.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git status --short
git add visual_deploy/ranking/target_ranker.py tests/test_target_ranker.py
git commit -m "feat: add target ranking"
```

---

### Task 10: Recorder And JSONL Artifacts

**Model routing:** `gpt-5.4`, medium effort.

**Files:**
- Create: `visual_deploy/recording/recorder.py`
- Test: `tests/test_recorder.py`

- [ ] **Step 1: Write failing recorder tests**

Create `tests/test_recorder.py`:

```python
import json

from visual_deploy.recording.recorder import RunRecorder


def test_recorder_writes_jsonl_records(tmp_path):
    rec = RunRecorder(tmp_path, run_name="test_run", config={"a": 1})
    rec.write_detection({"frame_id": 1, "detections": []})
    rec.write_candidate({"frame_id": 1, "track_id": 2})
    rec.write_target({"frame_id": 1, "valid": False, "reason": "no_valid_grasp_candidate"})
    assert (tmp_path / "test_run" / "run_config.yaml").exists()
    det = json.loads((tmp_path / "test_run" / "detections.jsonl").read_text(encoding="utf-8").splitlines()[0])
    target = json.loads((tmp_path / "test_run" / "targets.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert det["frame_id"] == 1
    assert target["reason"] == "no_valid_grasp_candidate"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_recorder.py -v
```

Expected: FAIL because recorder does not exist.

- [ ] **Step 3: Implement recorder**

Create `visual_deploy/recording/recorder.py`:

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


class RunRecorder:
    def __init__(self, output_root: str | Path, run_name: str | None = None, config: dict[str, Any] | None = None) -> None:
        name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = Path(output_root) / name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        for subdir in ("frames", "masks", "overlays"):
            (self.run_dir / subdir).mkdir(exist_ok=True)
        with (self.run_dir / "run_config.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(config or {}, f, allow_unicode=True, sort_keys=False)

    def _write_jsonl(self, filename: str, record: dict[str, Any]) -> None:
        with (self.run_dir / filename).open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_detection(self, record: dict[str, Any]) -> None:
        self._write_jsonl("detections.jsonl", record)

    def write_candidate(self, record: dict[str, Any]) -> None:
        self._write_jsonl("candidates.jsonl", record)

    def write_target(self, record: dict[str, Any]) -> None:
        self._write_jsonl("targets.jsonl", record)

    def write_error(self, record: dict[str, Any]) -> None:
        self._write_jsonl("errors.jsonl", record)
```

- [ ] **Step 4: Run recorder tests**

Run:

```powershell
python -m pytest tests/test_recorder.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

Run:

```powershell
git status --short
git add visual_deploy/recording/recorder.py tests/test_recorder.py
git commit -m "feat: add run recorder"
```

---

### Task 11: Offline Smoke Pipeline

**Model routing:** `gpt-5.5`, high effort for integration review.

**Files:**
- Create: `visual_deploy/pipeline/offline_pipeline.py`
- Create: `scripts/run_offline_smoke.py`
- Test: `tests/test_offline_pipeline.py`

- [ ] **Step 1: Write failing offline pipeline test with mocks**

Create `tests/test_offline_pipeline.py`:

```python
import numpy as np

from visual_deploy.detection.yolo_detector import MockDetector
from visual_deploy.pipeline.offline_pipeline import OfflinePipeline
from visual_deploy.types import CameraIntrinsics, DeployFrame


class MockSegmentor:
    def infer(self, color_bgr_256, depth_mm_256):
        from visual_deploy.segmentation.gcnet_segmentor import SegmentResult
        mask = np.zeros((256, 256), dtype=bool)
        yy, xx = np.ogrid[:256, :256]
        mask[(yy - 128) ** 2 + (xx - 128) ** 2 <= 60 ** 2] = True
        return SegmentResult(mask.astype(np.float32), mask, int(mask.sum()), 1.0)


def test_offline_pipeline_emits_target_with_mock_detector_segmentor(tmp_path):
    frame = DeployFrame(
        frame_id=1,
        timestamp_ms=1.0,
        color_bgr=np.zeros((480, 640, 3), dtype=np.uint8),
        depth_mm=np.full((480, 640), 500.0, dtype=np.float32),
        intrinsics=CameraIntrinsics(fx=600.0, fy=600.0, ppx=320.0, ppy=240.0, depth_scale=0.001),
    )
    pipeline = OfflinePipeline(detector=MockDetector(0.9), segmentor=MockSegmentor(), config={"recording": {"output_root": str(tmp_path)}})
    target = pipeline.process_frame(frame)
    assert target.valid is True
    assert target.z_mm is not None
    assert target.xyz_camera_m is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_offline_pipeline.py -v
```

Expected: FAIL because offline pipeline does not exist.

- [ ] **Step 3: Implement offline pipeline**

Create `visual_deploy/pipeline/offline_pipeline.py` with orchestration that:

```text
1. detector.infer(frame.color_bgr)
2. gate detections
3. update depth stats and tracker
4. skip non-confirmed tracks
5. build RoiTransform from track bbox
6. crop and resize BGR/depth to 256
7. update DepthFusionBuffer
8. segment current ROI
9. select_grasp_patch(mask, fused_depth, cam_roi)
10. rank candidates
11. return GraspTarget
```

Use `cv2.resize`:

```python
color_roi = cv2.resize(color_crop, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)
depth_roi = cv2.resize(depth_crop, (roi_size, roi_size), interpolation=cv2.INTER_NEAREST)
```

If no valid candidates exist, return:

```python
GraspTarget(valid=False, frame_id=frame.frame_id, reason="no_valid_grasp_candidate")
```

- [ ] **Step 4: Add offline smoke script**

Create `scripts/run_offline_smoke.py` that loads a single RGB image and `.npy` depth map, constructs `DeployFrame`, uses `MockDetector` if no YOLO weights are present, and prints the target JSON. Keep command-line arguments:

```text
--config configs/deploy.yaml
--rgb path/to/rgb.png
--depth path/to/depth.npy
--fx 600 --fy 600 --ppx 320 --ppy 240
```

- [ ] **Step 5: Run offline pipeline test**

Run:

```powershell
python -m pytest tests/test_offline_pipeline.py -v
```

Expected: PASS.

- [ ] **Step 6: Run accumulated focused tests**

Run:

```powershell
python -m pytest tests/test_geometry_roi.py tests/test_grasp_patch.py tests/test_depth_fusion.py tests/test_offline_pipeline.py -v
```

Expected: PASS.

- [ ] **Step 7: Commit**

Run:

```powershell
git status --short
git add visual_deploy/pipeline/offline_pipeline.py scripts/run_offline_smoke.py tests/test_offline_pipeline.py
git commit -m "feat: add offline RGB-D grasp pipeline"
```

---

### Task 12: RealSense Runtime Source

**Model routing:** `gpt-5.4`, medium effort; main agent reviews hardware assumptions.

**Files:**
- Create: `visual_deploy/camera/realsense_source.py`
- Create: `scripts/run_realtime.py`
- Test: `tests/test_realsense_source.py`

- [ ] **Step 1: Write import-safe RealSense test**

Create `tests/test_realsense_source.py`:

```python
from visual_deploy.camera.realsense_source import RealSenseUnavailableError


def test_realsense_unavailable_error_is_importable():
    err = RealSenseUnavailableError("missing")
    assert "missing" in str(err)
```

- [ ] **Step 2: Implement RealSense source**

Create `visual_deploy/camera/realsense_source.py`:

```python
from __future__ import annotations

import numpy as np

from visual_deploy.types import CameraIntrinsics, DeployFrame

try:
    import pyrealsense2 as rs
except Exception:
    rs = None


class RealSenseUnavailableError(RuntimeError):
    pass


class RealSenseSource:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        align_to_color: bool = True,
        spatial_filter: bool = True,
        temporal_filter: bool = False,
        hole_filling_filter: bool = False,
    ) -> None:
        if rs is None:
            raise RealSenseUnavailableError("pyrealsense2 is not installed")
        self.frame_id = 0
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color) if align_to_color else None
        self.spatial = rs.spatial_filter() if spatial_filter else None
        self.temporal = rs.temporal_filter() if temporal_filter else None
        self.hole = rs.hole_filling_filter() if hole_filling_filter else None
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale_m = float(depth_sensor.get_depth_scale())

    def __iter__(self) -> "RealSenseSource":
        return self

    def __next__(self) -> DeployFrame:
        frames = self.pipeline.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            raise StopIteration
        if self.spatial is not None:
            depth_frame = self.spatial.process(depth_frame)
        if self.temporal is not None:
            depth_frame = self.temporal.process(depth_frame)
        if self.hole is not None:
            depth_frame = self.hole.process(depth_frame)
        color_bgr = np.asanyarray(color_frame.get_data())
        depth_raw = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth_mm = depth_raw * self.depth_scale_m * 1000.0
        intr = color_frame.profile.as_video_stream_profile().intrinsics
        frame = DeployFrame(
            frame_id=self.frame_id,
            timestamp_ms=float(color_frame.get_timestamp()),
            color_bgr=color_bgr,
            depth_mm=depth_mm,
            intrinsics=CameraIntrinsics(float(intr.fx), float(intr.fy), float(intr.ppx), float(intr.ppy), 0.001),
            meta={"source": "realsense", "aligned": self.align is not None},
        )
        self.frame_id += 1
        return frame

    def close(self) -> None:
        self.pipeline.stop()
```

- [ ] **Step 3: Add realtime runner**

Create `scripts/run_realtime.py` to load `configs/deploy.yaml`, construct `RealSenseSource`, detector, segmentor, `OfflinePipeline` or `RealtimePipeline`, and print/write target results. Keep it synchronous.

- [ ] **Step 4: Run import-safe test**

Run:

```powershell
python -m pytest tests/test_realsense_source.py -v
```

Expected: PASS whether or not hardware is attached.

- [ ] **Step 5: Run hardware smoke when D435i is attached**

Run:

```powershell
python scripts/run_realtime.py --config configs/deploy.yaml --max-frames 30
```

Expected: reads 30 aligned frames and writes target records or structured invalid target records. If hardware is unavailable, record that this verification was skipped.

- [ ] **Step 6: Commit**

Run:

```powershell
git status --short
git add visual_deploy/camera/realsense_source.py scripts/run_realtime.py tests/test_realsense_source.py
git commit -m "feat: add RealSense runtime source"
```

---

### Task 13: Documentation And Final Verification

**Model routing:** main agent `gpt-5.5`, high effort.

**Files:**
- Modify: `README.md`
- Optional create: `docs/deployment.md`

- [ ] **Step 1: Update README**

Write README sections:

```text
1. Runtime purpose
2. Environment setup
3. Required weights
4. GCNet export command
5. YOLO weight requirement
6. Offline smoke command
7. Realtime command
8. Output record files
9. Known non-goals: robot extrinsics, IK, collision checks, thread optimization
```

- [ ] **Step 2: Run full test suite**

Run:

```powershell
python -m pytest -v
```

Expected: PASS. If hardware-dependent tests are skipped, state the skip reason.

- [ ] **Step 3: Run git status**

Run:

```powershell
git status --short
```

Expected: only README/doc files modified before final docs commit.

- [ ] **Step 4: Commit documentation**

Run:

```powershell
git add README.md docs/deployment.md
git commit -m "docs: add deployment usage guide"
```

Adjust `git add` if `docs/deployment.md` is not created.

- [ ] **Step 5: Final verification before completion**

Run:

```powershell
git status --short
git log --oneline -10
python -m pytest -v
```

Expected: clean or only intentionally untracked local weights/runs ignored by git, recent commits show small focused changes, tests pass.

---

## Scope Deferred Until After Basic Functionality

Do not implement these in the first pass:

- Threaded camera/detection/segmentation workers.
- Robot extrinsics, IK, collision, or workspace checks.
- Full-frame depth fusion.
- ByteTrack/Kalman replacement unless IoU tracker blocks functionality.
- Committing large model weights.

## Plan Self-Review

Spec coverage:

- Self-contained Visual-Deploy runtime: Tasks 1, 6, 8, 13.
- ROI and intrinsics correctness: Task 2.
- Patch grasp camera-layer pose: Task 3.
- Depth fusion for confirmed tracks: Task 5.
- YOLO and postprocess: Tasks 4 and 8.
- GCNet reparameterized export: Task 6.
- Candidate ranking and single target output: Tasks 9 and 11.
- Recording: Task 10.
- RealSense filtering: Task 12.
- Git commit discipline: every task has a commit step.

No implementation task should proceed without focused tests and a commit boundary.
