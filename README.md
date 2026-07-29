# Visual-Deploy 2.0

Visual-Deploy is a self-contained RealSense RGB-D pipeline that returns one
camera-frame apple grasp target. Version 2.0 runs detection and segmentation
from ONNX or device-built TensorRT engines. Training repositories and custom
training modules are not runtime dependencies.

## What changed in 2.0

- The detector is a standard YOLOv10n inference graph exported from the selected
  false-positive-sensitive SAM-YOLO checkpoint. ECA/CBAM and the training-only
  SAM auxiliary head are absent from the deployment graph.
- Both YOLOv10 and RGBD-GCNet use ONNX Runtime on a PC. A shared TensorRT engine
  session and a Jetson-side `trtexec` build script are included.
- The validated dual-threshold tracker is used: high `0.50`, low `0.10`, IoU
  `0.30`, `min_hits=1`, and `max_lost=30` at 30 FPS.
- Timing, debug snapshots, diagnostics, JSONL logs, and artifacts have explicit
  switches and are off by default. Camera-layer safety validation remains on.
- Runtime telemetry now reserves capture, stage, queue, dropped-frame, frame-age,
  and device-resource fields for later bounded-queue threading work.

## Runtime pipeline

```text
aligned D435i color + depth_mm
  -> standard YOLOv10 ONNX/TensorRT
  -> dual-threshold IoU association
  -> ROI-remapped per-track depth fusion
  -> RGBD-GCNet ONNX/TensorRT, fixed 256 x 256 ROI
  -> suction patch geometry and ranking
  -> static safety and per-track continuity validation
  -> GraspTarget in the camera frame
```

Depth remains in millimeters until backprojection. `depth_scale=0.001` converts
the selected depth to meters for `xyz_camera_m`.

## Model artifacts

| Artifact | Contract | SHA-256 |
|---|---|---|
| `weights/yolov10_sam_robust_standard.onnx` | `images [1,3,640,640] -> output0 [1,300,6]` | `3632ee3f...68a5330` |
| `weights/rgbd_gcnet_l03_robustft.onnx` | `bgr [1,3,256,256] + depth_mm [1,1,256,256] -> prob [1,1,256,256]` | `a9091738...d4390330` |

The exact source paths, full hashes, input/output shapes, and export checks are
stored beside each model in `.onnx.json` files.

The detector source is:

```text
D:\Github Code\yolov10-improved\runs\sam_negative_mixed_finetune\
seed2_lr2e-4_e5_source75_negative25\weights\best.pt
```

The deployment setting is `detection.conf_threshold: 0.50`. The selected model
keeps apple validation mAP50-95 at `0.74469` and reduced held-out negative boxes
by `93.9%` at confidence 0.50 in its selection audit.

## PC installation

Use Python 3.10 or newer:

```powershell
cd "D:\Github Code\Visual-Deploy"
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

`requirements-export.txt` is only for a development machine that must recreate
the ONNX files. It is not needed at runtime.

## Run

Offline RGB-D check:

```powershell
.\.venv\Scripts\python.exe scripts\run_offline_smoke.py `
  --config configs\deploy.cpu.local.yaml `
  --rgb samples\rgb.png --depth samples\depth_mm.npy `
  --fx 600 --fy 600 --ppx 320 --ppy 240
```

The four intrinsics above are examples only. Use the intrinsics stored with the
offline frame. Live mode reads the color-stream intrinsics from the D435i.

Live D435i:

```powershell
.\.venv\Scripts\python.exe scripts\run_realtime.py --config configs\deploy.yaml
```

The production config uses TensorRT engines and enables a capacity-one latest-frame
capture thread. Stale camera frames are replaced instead of accumulating latency;
tracking, depth fusion, YOLO, GCNet, and safety still run serially in one owner
thread. Use `--sync` to restore the legacy synchronous loop for comparison. Use
`configs/deploy.cpu.local.yaml` for the portable CPU/ONNX path.

Sparse debug viewer:

```powershell
.\.venv\Scripts\python.exe scripts\run_debug_viewer.py --config configs\deploy.yaml
```

Viewer keys are `q`/`Esc` quit, `p` pause, `m` mask, `d` depth, and `s` snapshot.
The viewer enables only the in-memory debug snapshot it needs.

## Runtime switches

Production defaults do not create a `runs/` directory.

```yaml
profiling:
  enabled: false
debug:
  enabled: false
diagnostics:
  enabled: false
recording:
  enabled: false
  save_detections: false
  save_candidates: false
  save_targets: false
  save_errors: false
  save_timings: false
  save_events: false
  save_event_artifacts: false
```

Enable `recording.enabled` plus only the channels needed for a specific test.
`profiling.enabled` adds `timings.jsonl`; `diagnostics.enabled` permits configured
safety/continuity events and artifacts.

For multi-apple scenes, the production config keeps the last valid target as the
active target. Ordinary frames run the expensive depth/segmentation/grasp branch
only for that track; failure triggers all remaining confirmed tracks in the same
frame, and every 30 processed frames restores full ranking:

```yaml
runtime:
  threaded_capture: true
  active_target:
    enabled: true
    refresh_interval_frames: 30
```

Set `active_target.enabled: false` to recover the original all-target-per-frame
behavior. Profiling exposes fast-path, fallback, refresh, and deferred-track
counts under `timings.workload`.

## Grasp candidate search

Deployment uses a deterministic coarse-to-fine search that pre-ranks candidates
with existing low-cost depth-support, boundary, and centroid terms, then runs the
full float64 plane, median, MAD, and final-score calculation on the best 256:

```yaml
grasp:
  candidate_top_k: 256
  exhaustive_fallback: true
  shadow_verify_every_n_frames: 0
```

Set `candidate_top_k: null` for exhaustive search. With fallback enabled, an
empty or below-score Top-K result is retried exhaustively before the frame is
rejected. Shadow verification is off in production; set its interval to `30`
during commissioning to return the exhaustive result periodically and record
pixel, depth, normal, and exact-match evidence in `candidates.jsonl`.

## Automated threading profile collection

The collector creates a temporary timing-only config, runs the existing offline
or realtime entry point, samples process CPU/RSS, starts `tegrastats` automatically
when it is available, and writes JSONL, JSON, CSV, and Markdown reports.

Laptop/offline example:

```powershell
.\.venv\Scripts\python.exe scripts\collect_thread_profile.py `
  --mode offline --config configs\deploy.cpu.local.yaml `
  --rgb samples\rgb.png --depth samples\depth_mm.npy `
  --fx 600 --fy 600 --ppx 320 --ppy 240 `
  --frames 300 --warmup-frames 20 --tegrastats off
```

Jetson/realtime example:

```bash
python scripts/collect_thread_profile.py \
  --mode realtime --config configs/deploy.yaml \
  --frames 300 --warmup-frames 20 \
  --session-name threaded_300
```

Use the same collector with `--sync` for a directly comparable legacy run:

```bash
python scripts/collect_thread_profile.py \
  --mode realtime --config configs/deploy.yaml \
  --frames 300 --warmup-frames 20 \
  --session-name sync_300 --sync
```

Each session is created under `runs/thread_profiles/` and contains `report.md`,
`summary.json`, `summary.csv`, raw process/tegrastats JSONL, child logs, the
generated config, and the pipeline `timings.jsonl`. Realtime runs populate the
capacity-one capture queue wait/depth/drop fields; offline runs leave queue fields
empty. `configs/deploy.yaml` is the Jetson/TensorRT production config.

Depth profiling now separates `depth_remap_ms` and `depth_median_ms`, and records
`depth_history_frames` plus `depth_remap_calls` under `timings.workload`. These
fields are emitted only through the existing profiling path; production defaults
remain free of timing records.

Threading overlaps only camera capture with the still-serialized perception
pipeline. In the representative Orin report, capture was about 16 ms and
perception was about 85 ms, so the synchronous cycle was roughly 101 ms while
the ideal overlapped cycle remains about 85 ms. The corresponding upper bound is
approximately 9.9 to 11.8 FPS, not 30 FPS. The capacity-one queue is primarily a
freshness and bounded-memory feature; dropped stale frames do not make YOLO,
depth fusion, GCNet, or grasp geometry execute faster.

Incomplete timing, child failure, timeout, interruption, or a failed requested
tegrastats stream returns a typed nonzero exit status and remains recorded in
`manifest.json`; these runs are never reported as complete.

## Jetson Orin NX quick path

1. Install a matching JetPack 6.2.x stack and verify TensorRT and CUDA.
2. Install and verify the D435i with `realsense-viewer`.
3. Install the NVIDIA PyTorch wheel that matches that exact JetPack release.
4. Create a virtual environment with system packages visible, then install
   `requirements-jetson.txt` and this project with `--no-deps`.
5. Copy the ONNX files to the Jetson and build `.engine` files on that Jetson:

```bash
python3 scripts/build_tensorrt_engines.py \
  --trtexec /usr/src/tensorrt/bin/trtexec
```

6. Verify the engine paths in `configs/deploy.yaml`; both production backends are
   already set to `tensorrt`.
7. Run a fixed offline RGB-D check before connecting the realtime camera.

TensorRT engines must not be copied between different TensorRT/JetPack/device
stacks. Full commands and checks are in [docs/deployment.md](docs/deployment.md).

## Verification

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current verified result is reported with each release commit; rerun the command
above after changing a model, TensorRT engine, or JetPack environment.

The release-confidence evidence and the mandatory Jetson device acceptance gate
are recorded in
[docs/release_confidence_audit_20260726.md](docs/release_confidence_audit_20260726.md).

## Safety and threading scope

The camera-layer safety line is closed for the current target contract. It does
not include robot extrinsics, workspace checks, IK, collision avoidance, actuator
limits, emergency stop, or hardware interlocks. See
[docs/deployment_v2_runtime_audit.md](docs/deployment_v2_runtime_audit.md) for the
exact closure boundary and the remaining Orin/threading measurements.
