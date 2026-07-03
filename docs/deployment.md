# Deployment Guide

This guide covers moving `Visual-Deploy` to another Windows PC and running the
RGB-D grasp pipeline without importing sibling research repositories.

## 1. Copy Folder

Copy the `Visual-Deploy` folder to the target PC. Keep these paths:

```text
Visual-Deploy/
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

Do not copy generated folders such as `runs/`, `.venv/`, `.pytest_cache/`, or
`__pycache__/`.

## 2. Create Environment

Use Python 3.10.

```powershell
cd "D:\Deploy\Visual-Deploy"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

`requirements.txt` installs the vendored YOLOv10-compatible Ultralytics wheel:

```text
vendor/ultralytics-8.1.34-py3-none-any.whl
```

Do not replace it with PyPI `ultralytics==8.1.34`; the PyPI wheel does not
export `YOLOv10`.

For GPU inference, install a PyTorch build matching the target CUDA driver
before running the requirements command. The exported GCNet artifact was
validated with `torch 2.0.1+cu118`.

## 3. Install RealSense Runtime

For realtime D435i use, install the Intel RealSense SDK on the target PC, then
install Python dependencies from `requirements.txt`. The Python package is:

```text
pyrealsense2
```

The runtime requires `camera.align_to_color: true` because detections and grasp
targets are color-frame pixels.

The current RealSense source enforces this contract. It calls
`rs.align(rs.stream.color)`, reads the aligned depth frame, and converts depth
to millimeters before passing RGB-D data to the pipeline.

## 4. Place Weights

Expected files:

```text
weights/yolo_detect.pt
weights/rgbd_gcnet_l03_robustft_inference.pt
```

Validated source weights:

```text
yolo_detect.pt
  D:\Github Code\yolov10-improved\runs\apple\04_combinations\combo_cbam_eca_e150_fresh_direct\weights\best.pt

rgbd_gcnet_l03_robustft_inference.pt
  exported from D:\Github Code\RGBD-GCNet\work_dirs\apple_roi\finetune\l03_apple_robust_aug_train_only_20260626\best_mIoU_iter_3000.pth
```

Weights are intentionally ignored by Git.

## 5. Runtime Configuration

Default config:

```text
configs/deploy.yaml
```

Important contracts:

- `detection.model_type: yolov10`
- `segmentation.depth_input_unit: mm`
- `camera.align_to_color: true`
- GCNet ROI input is fixed at `256 x 256`
- Color ROI resize uses bilinear interpolation
- Depth ROI resize uses nearest-neighbor interpolation

Depth is stored in millimeters inside the runtime. Camera coordinates use:

```text
depth_m = depth_mm * CameraIntrinsics.depth_scale
CameraIntrinsics.depth_scale = 0.001
```

## 6. Offline Verification

Use real weights:

```powershell
.\.venv\Scripts\python.exe scripts\run_offline_smoke.py `
  --config configs\deploy.yaml `
  --rgb samples\rgb.png `
  --depth samples\depth_mm.npy `
  --fx 600 --fy 600 --ppx 320 --ppy 240
```

Use mock models only for plumbing checks:

```powershell
.\.venv\Scripts\python.exe scripts\run_offline_smoke.py `
  --config configs\deploy.yaml `
  --rgb samples\rgb.png `
  --depth samples\depth_mm.npy `
  --fx 600 --fy 600 --ppx 320 --ppy 240 `
  --use-mock-models
```

## 7. Realtime Run

```powershell
.\.venv\Scripts\python.exe scripts\run_realtime.py --config configs\deploy.yaml --max-frames 30
```

The command creates a run directory under `runs/` and prints target records.

## 8. Debug Viewer

For visual checks with D435i:

```powershell
.\.venv\Scripts\python.exe scripts\run_debug_viewer.py --config configs\deploy.yaml
```

The viewer keeps the main RGB image clean: detection boxes, optional mask tint,
and the selected grasp point. Target values and controls are shown in a bottom
status bar, while terminal JSON and `runs/` artifacts keep the detailed
records.

Keys:

```text
q / Esc  quit
p        pause/resume
m        show/hide mask
d        show/hide depth preview
s        save current debug snapshot
```

Small in-image labels are off by default. Add `--labels` when frame-level id,
confidence, or target depth labels are useful.

Offline visual check:

```powershell
.\.venv\Scripts\python.exe scripts\run_debug_viewer.py `
  --config configs\deploy.yaml `
  --rgb samples\rgb.png `
  --depth samples\depth_mm.npy `
  --fx 600 --fy 600 --ppx 320 --ppy 240 `
  --repeat-frames 30
```

## 9. Output Contract

For a valid target:

```text
u_px, v_px       Full-frame color pixel
z_mm             Selected depth in millimeters
xyz_camera_m     Camera-frame 3D point in meters
normal_xyz       Local patch normal
approach_axis    Suction approach axis, equal to -normal_xyz
target_score     Ranking score
```

The repository does not include robot extrinsics, IK, collision checks, or
workspace checks.

## 10. Test Suite

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Validated result:

```text
133 passed, 1 skipped
```

Hardware and real-scene realtime tests are separate from the unit test suite.
