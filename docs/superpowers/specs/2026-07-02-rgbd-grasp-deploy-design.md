# RGB-D Grasp Deployment Design

## Goal

Build `Visual-Deploy` into a lightweight, self-contained RGB-D grasp deployment project. The deployed folder must be movable to another computer, where it can run after installing its own environment, placing exported weights, and connecting an Intel RealSense D435i.

The runtime pipeline is:

```text
RealSense D435i RGB-D
  -> YOLO detection
  -> multi-frame detection postprocess
  -> per-track ROI depth fusion
  -> RGBD-GCNet l03 robustFT segmentation
  -> suction patch grasp selection
  -> one main camera-frame grasp target
```

The project must record intermediate YOLO boxes and masks, but only output one main grasp target per frame.

## Delivery Boundary

`Visual-Deploy` is not an import wrapper over the existing research repositories. During development, the other repositories may be used to export weights, compare outputs, and validate behavior. The final runtime must not require these repositories to exist:

```text
D:\Github Code\yolov10-improved
D:\Github Code\yolo-multi-frame-postprocess
D:\Github Code\RGBD-GCNet
```

Deployment dependencies and minimal runtime code live inside `Visual-Deploy`. Training code, experiment logs, full datasets, and research-only utilities stay out of the runtime package.

## Proposed Layout

```text
Visual-Deploy/
  README.md
  pyproject.toml or requirements.txt
  configs/
    deploy.yaml
  weights/
    yolo_detect.pt
    rgbd_gcnet_l03_robustft_inference.pt
  visual_deploy/
    camera/
      realsense_source.py
      realsense_filters.py
    detection/
      yolo_detector.py
    tracking/
      tracker.py
      depth_fusion.py
    segmentation/
      gcnet_segmentor.py
    geometry/
      roi.py
      backproject.py
      grasp_patch.py
      pose.py
    ranking/
      target_ranker.py
    recording/
      recorder.py
    pipeline/
      realtime_pipeline.py
  scripts/
    run_realtime.py
    run_offline_smoke.py
    verify_mapping.py
  tests/
```

The implementation may port small, stable runtime functions from the existing repositories, such as ROI transforms, backprojection, and patch selection. It must not pull in large training stacks.

## Runtime Data Flow

The first version uses a synchronous loop for correctness before thread optimization:

```text
for frame in RealSense:
    read color_bgr, depth_mm, intrinsics
    apply RealSense depth filters
    run YOLO detector
    update detection gate + tracker
    for each confirmed track:
        build training-matched ROI
        crop/resize color and depth to 256
        update per-track depth fusion buffer
        run GCNet mask inference on current ROI depth
        run patch grasp on fused ROI depth
        record candidate artifacts and metrics
    rank valid candidates
    output one main target or structured invalid result
```

Tentative tracks are recorded but do not enter segmentation or grasp. Confirmed tracks all run segmentation and grasp so that rejected candidates remain auditable.

## Core Data Contracts

```text
DeployFrame:
  frame_id
  timestamp_ms
  color_bgr
  depth_mm
  intrinsics

Track:
  track_id
  bbox_xyxy
  class_id
  confidence
  weighted_confidence
  state
  hits
  lost
  depth_stats

RoiPacket:
  track_id
  roi_transform
  color_bgr_256
  depth_mm_256
  cam_roi

SegmentResult:
  track_id
  prob_256
  mask_256
  mask_area
  largest_component_ratio

GraspCandidate:
  track_id
  u_roi
  v_roi
  u_img
  v_img
  z_mm
  xyz_camera_m
  normal_xyz
  approach_axis
  grasp_score
  target_score
  reject_reason
```

The output grasp pose is camera-layer only:

```text
position_m = [X, Y, Z]
normal_xyz = local surface normal from depth-plane fit
approach_axis = -normal_xyz
roll_free = true
```

No robot extrinsics, IK, collision, or workspace checks are included in this phase.

## ROI And Camera Geometry

Deployment must match the existing training ROI generation:

```text
bbox -> square_pad_bbox(bbox, pad_ratio=0.20, image_w, image_h)
crop -> resize to 256 x 256
RGB resize: bilinear
depth resize: nearest
mask resize: nearest
```

The crop may be clipped near image boundaries, so `scale_x` and `scale_y` must be tracked independently:

```text
scale_x = 256 / (x1 - x0)
scale_y = 256 / (y1 - y0)
```

ROI intrinsics:

```text
fx_roi = fx * scale_x
fy_roi = fy * scale_y
ppx_roi = (ppx - x0) * scale_x
ppy_roi = (ppy - y0) * scale_y
depth_scale = 0.001
```

Coordinate mapping:

```text
u_roi = (u_img - x0) * scale_x
v_roi = (v_img - y0) * scale_y

u_img = u_roi / scale_x + x0
v_img = v_roi / scale_y + y0
```

Patch selection runs in ROI coordinates. Final `u_img, v_img, z_mm` are backprojected using full-frame intrinsics:

```text
X = (u_img - ppx) * Z / fx
Y = (v_img - ppy) * Z / fy
Z = z_mm * 0.001
```

## Depth Policy

Runtime storage and geometry use millimeters:

```text
depth_mm
depth_roi_256_mm
fused_depth_roi_256_mm
```

GCNet training used depth in meters, so the exported inference wrapper must convert internally:

```text
depth_m = depth_mm * 0.001
```

Depth usage split:

```text
YOLO depth stats: current filtered depth_mm
GCNet segmentation: current filtered depth_roi_256_mm
patch grasp: fused_depth_roi_256_mm
```

This keeps segmentation aligned with the current RGB frame while giving patch grasp a more stable local surface.

## RealSense Filtering

The camera path uses aligned depth-to-color frames:

```text
color/depth stream
  -> align depth to color
  -> spatial_filter
  -> optional temporal_filter
  -> optional hole_filling_filter
  -> depth_mm float32
```

Default:

```yaml
spatial_filter: true
temporal_filter: false
hole_filling_filter: false
```

Temporal filtering is not enabled by default because per-track ROI depth fusion already handles short-window smoothing, and stacked temporal filters can create dynamic-object lag.

## DepthFusionBuffer

All confirmed tracks maintain ROI-level depth history:

```text
TrackDepthEntry:
  frame_id
  timestamp_ms
  roi_transform
  depth_roi_256_mm
  valid_mask_256
```

Fusion output:

```text
FusedTrackDepth:
  track_id
  depth_roi_256_mm
  valid_ratio
  source_frame_count
  age_ms
```

First implementation:

```text
window_size = 5
valid depth = finite and min_depth_mm <= depth <= max_depth_mm
fusion = per-pixel median
fallback = current depth ROI when insufficient valid history exists
```

Each historical depth crop is stored after transformation to its own `256x256` ROI coordinate system. Full-frame depth maps are not fused in the first version.

## GCNet Inference Export

The GCNet export is not a plain checkpoint save. It must follow the network's deploy-time design:

```text
l03 robustFT checkpoint
  -> build original mmseg model
  -> load checkpoint
  -> model.backbone.switch_to_deploy()
  -> wrap BGR + depth_mm inference preprocessing
  -> export inference .pt
```

The improved backbone is `RGBDGCNetCMEAFM`:

```text
input x: [RGB, depth]
rgb_stem(rgb)
depth_stem(raw_depth)
CME(depth_feat, raw_depth) -> depth confidence
depth_feat *= confidence
depth_branch_layers at stages 4/5/6
AFM fuses depth features into detail/RGB features at stages 4/5/6
decode_head outputs segmentation logits in eval mode
```

Deploy-time reparameterization applies to `GCBlock` instances in:

```text
rgb_stem
depth_stem
semantic_branch_layers
detail_branch_layers
depth_branch_layers
```

Each train-time `GCBlock` has multiple paths:

```text
path_3x3_1 + path_3x3_2 + path_1x1 + residual BN
```

`switch_to_deploy()` fuses those paths into one `reparam_3x3 Conv2d`, which is the lossless lightweight inference form.

These modules are retained as normal inference modules in phase 1:

```text
CME ConvModule stack
AFM depth_project/channel_attention/spatial_attention/out_project
DAPPM
decode_head
ordinary ConvModule compression/downsample layers
```

Depth handling must be documented explicitly:

```text
CME resizes raw_depth to depth_feat resolution before confidence estimation.
AFM resizes depth_feat to RGB/detail feature resolution before attention fusion.
```

The exported wrapper receives BGR and depth in millimeters:

```text
input:
  color_bgr_256
  depth_mm_256

internal:
  BGR -> RGB
  depth_mm -> depth_m
  RGB/depth preprocessing
  backbone + decode_head
  softmax + threshold

output:
  foreground_prob_256
  mask_256
```

Required GCNet export checks:

```text
1. original eval model vs switch_to_deploy model
2. switch_to_deploy model vs exported inference wrapper
3. expected depth unit behavior with known depth input
```

Compare logits/probability MAE, mask IoU, and foreground area. Mask IoU should be near exact for fixed test inputs.

## YOLO Deployment

YOLO must also be self-contained for the final package. The first implementation must test whether `weights/yolo_detect.pt` can be loaded in the clean `Visual-Deploy` environment without importing the `yolov10-improved` repository.

If direct loading fails, the acceptable fallback is to export a self-contained inference format or vendor the minimum runtime code. Copying the full training repository is not acceptable for the deployment package.

## Detection Postprocess

The postprocess logic is ported into `Visual-Deploy` as minimal runtime modules:

```text
ConfidenceGate
SimpleIoUTracker
DepthRoiStats
TrackWindowBuffer
```

Default tracking policy:

```yaml
high_conf_threshold: 0.7
low_conf_threshold: 0.4
iou_threshold: 0.3
min_hits: 2
max_lost: 30
```

This can later be replaced by ByteTrack/Kalman after the RGB-D grasp loop is verified.

## Target Ranking

All confirmed tracks are processed and recorded. Only valid grasp candidates participate in target selection.

Initial configurable score:

```text
target_score =
  0.45 * grasp_score
+ 0.20 * depth_valid_score
+ 0.15 * track_confidence
+ 0.10 * track_stability
+ 0.10 * mask_quality_score
```

Reject reasons are recorded for invalid candidates:

```text
no_mask
mask_area_too_small
no_grasp_patch
grasp_score_below_threshold
depth_invalid
```

## Recording

Each run creates:

```text
runs/YYYYMMDD_HHMMSS/
  run_config.yaml
  frames/
  masks/
  overlays/
  detections.jsonl
  candidates.jsonl
  targets.jsonl
  errors.jsonl
```

`detections.jsonl` records raw detections and tracks. `candidates.jsonl` records every confirmed-track segmentation and grasp attempt. `targets.jsonl` records the single main output target per frame.

Invalid target output is structured:

```json
{"valid": false, "reason": "no_valid_grasp_candidate"}
```

Valid target output:

```json
{
  "valid": true,
  "track_id": 3,
  "u_px": 642.5,
  "v_px": 358.0,
  "z_mm": 512.3,
  "xyz_camera_m": [0.001, -0.002, 0.512],
  "approach_axis": [0.02, 0.01, 0.999],
  "target_score": 0.86
}
```

## Configuration

`configs/deploy.yaml` should include:

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

## Verification Plan

Required tests before claiming the deployment loop works:

```text
1. ROI coordinate round-trip
2. edge-clipped ROI with scale_x != scale_y
3. ROI/full-frame intrinsics backprojection consistency
4. depth_mm to z_m unit test
5. GCNet original eval vs switch_to_deploy output consistency
6. GCNet switch_to_deploy vs exported wrapper consistency
7. YOLO weight clean-environment load smoke test
8. DepthFusionBuffer median and fallback behavior
9. confirmed tracks only enter segmentation/grasp
10. target ranking selects expected main target
11. offline RGB-D smoke pipeline emits detections/candidates/target records
12. RealSense smoke reads aligned depth with spatial filter enabled
```

The most important geometry invariant is:

```text
backproject_roi(u_roi, v_roi, z, cam_roi)
approximately equals
backproject_full(u_img, v_img, z, cam_full)
```

## Implementation Work Splitting

Use subagents only for independent domains. The main agent keeps architecture and final correctness decisions.

```text
Subagent A:
  model: gpt-5.4-mini
  effort: low
  scope: read-only YOLO weight loading and dependency extraction

Subagent B:
  model: gpt-5.4
  effort: medium
  scope: Visual-Deploy ROI, backprojection, and depth fusion tests/modules

Subagent C:
  model: gpt-5.5
  effort: high
  scope: GCNet l03 robustFT reparameterized inference export

Subagent D:
  model: gpt-5.4
  effort: medium
  scope: recorder, config schema, and offline smoke runner

Main agent:
  model: gpt-5.5
  effort: high
  scope: integration, mapping review, end-to-end verification, commit coordination
```

## Git Discipline

Implementation should use small, reviewable commits:

```text
1. design/spec commit
2. project scaffold and config commit
3. geometry and mapping tests commit
4. depth fusion/tracking commit
5. GCNet export commit
6. segmentation loader commit
7. YOLO loader and postprocess commit
8. recorder and offline smoke commit
9. realtime RealSense integration commit
10. final verification/documentation commit
```

Before each commit:

```text
git status --short
run focused tests for touched modules
avoid staging unrelated user changes
```

Do not rewrite or reset unrelated worktree changes.
