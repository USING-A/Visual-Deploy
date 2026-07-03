# Visual-Deploy

Visual-Deploy is a self-contained RGB-D deployment pipeline for apple grasp
target generation. It acquires aligned RGB-D frames, detects apples, stabilizes
detections across frames, segments each confirmed ROI, selects a suction grasp
patch, records intermediate results, and returns a camera-frame grasp target.

Runtime code imports only this repository, installed Python packages, local
configuration, and local weights. The research repositories are used only for
model training/export and are not runtime dependencies.

## Features

- RealSense D435i RGB-D source with depth aligned to color.
- YOLOv10 apple detector loaded from exported/local `.pt` weights.
- Confidence gating, IoU tracking, and per-track ROI depth fusion.
- Reparameterized RGBD-GCNet TorchScript segmentation.
- ROI-safe geometry mapping from detector box to 256x256 GCNet input and back
  to full-frame pixels.
- Suction grasp patch selection and camera-frame `(u, v, z_mm, xyz, pose)`
  output.
- JSONL recording of detections, candidates, targets, and errors.

## Pipeline

```text
RealSense D435i RGB-D frame
  -> YOLOv10 detection
  -> confidence gate and IoU tracker
  -> per-track ROI crop and depth fusion
  -> exported RGBD-GCNet segmentation
  -> suction grasp patch selection
  -> target ranking
  -> GraspTarget
```

Depth is carried in millimeters inside the runtime. Camera backprojection uses
`CameraIntrinsics.depth_scale=0.001`, so `depth_mm * 0.001 = meters`.

## Project Layout

```text
configs/deploy.yaml                 Runtime configuration
requirements.txt                    Deployment environment requirements
vendor/                             Vendored YOLOv10-compatible wheel
weights/                            Local model weights, ignored by Git
scripts/export_gcnet_l03_inference.py
scripts/run_offline_smoke.py
scripts/run_realtime.py
visual_deploy/                      Runtime package
tests/                              Unit and smoke tests
docs/deployment.md                  Detailed deployment notes
```

## Installation

Use Python 3.10 on the deployment PC.

```powershell
cd "D:\Github Code\Visual-Deploy"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

For a CUDA deployment, install the PyTorch build that matches the target GPU
and driver first, then run the same requirements command. The validated export
environment used `torch 2.0.1+cu118`.

## Required Weights

Create `weights/` and place:

```text
weights/yolo_detect.pt
weights/rgbd_gcnet_l03_robustft_inference.pt
```

Current validated sources:

```text
yolo_detect.pt
  D:\Github Code\yolov10-improved\runs\apple\04_combinations\combo_cbam_eca_e150_fresh_direct\weights\best.pt

rgbd_gcnet_l03_robustft_inference.pt
  exported from D:\Github Code\RGBD-GCNet\work_dirs\apple_roi\finetune\l03_apple_robust_aug_train_only_20260626\best_mIoU_iter_3000.pth
```

The detector config must keep:

```yaml
detection:
  model_type: yolov10
```

PyPI `ultralytics==8.1.34` does not provide `YOLOv10`. This project vendors
the YOLOv10-compatible `ultralytics` wheel built from `yolov10-improved`.

## GCNet Export

Run from the Visual-Deploy root when the RGBD-GCNet development environment is
available:

```powershell
$env:PYTHONUTF8='1'
D:\Github Code\RGBD-GCNet\.venv\Scripts\python.exe scripts\export_gcnet_l03_inference.py `
  --config "D:\Github Code\RGBD-GCNet\configs\fuji\seg\rgbd_gcnet_s_roi_cme_afm_boundary_l03_apple_robust_aug_train_only.py" `
  --checkpoint "D:\Github Code\RGBD-GCNet\work_dirs\apple_roi\finetune\l03_apple_robust_aug_train_only_20260626\best_mIoU_iter_3000.pth" `
  --output weights\rgbd_gcnet_l03_robustft_inference.pt `
  --device cpu `
  --rgbd-root "D:\Github Code\RGBD-GCNet"
```

The export calls the improved GCNet `switch_to_deploy()` path so training-time
heavy GC blocks are reparameterized before TorchScript tracing.

## Offline Smoke

Real weights:

```powershell
.\.venv\Scripts\python.exe scripts\run_offline_smoke.py `
  --config configs\deploy.yaml `
  --rgb samples\rgb.png `
  --depth samples\depth_mm.npy `
  --fx 600 --fy 600 --ppx 320 --ppy 240
```

Pipeline-only mock mode:

```powershell
.\.venv\Scripts\python.exe scripts\run_offline_smoke.py `
  --config configs\deploy.yaml `
  --rgb samples\rgb.png `
  --depth samples\depth_mm.npy `
  --fx 600 --fy 600 --ppx 320 --ppy 240 `
  --use-mock-models
```

Mock mode is explicit. Missing real weights fail loudly by default.

## Realtime Run

With D435i connected and both weights present:

```powershell
.\.venv\Scripts\python.exe scripts\run_realtime.py --config configs\deploy.yaml --max-frames 30
```

`camera.align_to_color` must remain `true`; detections and grasp points are
color-frame pixels.

## Outputs

Each run creates a timestamped directory under `runs/`:

```text
run_config.yaml
detections.jsonl
candidates.jsonl
targets.jsonl
errors.jsonl
frames/
masks/
overlays/
```

For a valid target:

```text
u_px, v_px       Full-frame color pixel
z_mm             Selected patch depth in millimeters
xyz_camera_m     Camera-frame point in meters
normal_xyz       Local patch normal
approach_axis    Suction approach direction, equal to -normal_xyz
target_score     Weighted ranking score
```

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current validated result:

```text
133 passed, 1 skipped
```

The skipped test is the optional YOLO weight smoke when weights or optional
runtime dependencies are unavailable.

## Deployment Copy Checklist

Copy this folder to the target PC with:

```text
configs/
docs/
scripts/
vendor/
visual_deploy/
weights/
README.md
requirements.txt
pyproject.toml
```

Do not copy `runs/`, `.venv/`, `.pytest_cache/`, or `__pycache__/`.

## Non-Goals

This repository does not perform robot extrinsics, robot workspace validation,
IK, collision checking, or threaded worker optimization. It returns the
camera-layer grasp target and suction approach direction only.
