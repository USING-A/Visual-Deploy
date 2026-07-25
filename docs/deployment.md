# Deployment guide

## 1. Copy the release

Copy these paths to the target machine:

```text
configs/
docs/
scripts/
visual_deploy/
weights/*.onnx
weights/*.onnx.json
README.md
requirements.txt
requirements-jetson.txt
pyproject.toml
```

Do not copy `runs/`, `.venv/`, `.pytest_cache/`, `outputs/`, or a TensorRT engine
built on another machine.

## 2. Model and coordinate contracts

The runtime accepts only `.onnx` or `.engine` model files.

```text
YOLOv10 input       RGB float32 NCHW, 1 x 3 x 640 x 640
YOLOv10 output      1 x 300 x 6, xyxy + confidence + class id
GCNet color input   BGR float32 NCHW, 1 x 3 x 256 x 256
GCNet depth input   float32 millimeters, 1 x 1 x 256 x 256
GCNet output        foreground probability, 1 x 1 x 256 x 256
mask                prob >= segmentation.threshold
```

Live depth is aligned to the color stream by `rs.align(rs.stream.color)`. Color
pixels and aligned depth therefore share the same image coordinates. Full-frame
camera intrinsics are adjusted only inside the cropped/resized ROI and the final
target is mapped back before backprojection.

## 3. PC setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q
```

Use `configs/deploy.cpu.local.yaml` for a CPU-only functional check. An ONNX
Runtime installation without `CUDAExecutionProvider` automatically uses CPU.

## 4. Recreate the ONNX artifacts

This is needed only when the selected checkpoints change. Use the validated
development environment with `torch`, `onnx`, `onnxruntime`, and the local
YOLOv10-compatible Ultralytics code.

```powershell
python scripts\export_yolov10_onnx.py `
  --source-repo "D:\Github Code\yolov10-improved" `
  --checkpoint "D:\Github Code\yolov10-improved\runs\sam_negative_mixed_finetune\seed2_lr2e-4_e5_source75_negative25\weights\best.pt" `
  --output weights\yolov10_sam_robust_standard.onnx

python scripts\export_gcnet_onnx.py `
  --torchscript weights\rgbd_gcnet_l03_robustft_inference.pt `
  --output weights\rgbd_gcnet_l03_robustft.onnx
```

The YOLO exporter asserts a `YOLOv10DetectionModel`/`v10Detect` graph, removes
the training-only SAM ROI distiller, rejects ECA modules, and runs an ONNX Runtime
smoke test. The GCNet exporter extracts a probability-only graph and verifies it
against TorchScript with a maximum absolute error limit of `1e-3`.

## 5. Jetson Orin NX setup

### 5.1 Install and inspect JetPack

Use NVIDIA SDK Manager or install the JetPack packages over a compatible Jetson
Linux image:

```bash
sudo apt update
sudo apt install nvidia-jetpack python3-venv python3-pip python3-opencv
cat /etc/nv_tegra_release
apt list --installed 2>/dev/null | grep -E 'nvidia-jetpack|tensorrt|libnvinfer'
/usr/src/tensorrt/bin/trtexec --version
```

The documented baseline is JetPack 6.2.x on Orin NX. Record the actual Jetson
Linux, CUDA, TensorRT, Python, and board model in the deployment log. Do not mix
wheels or engines from another JetPack line.

### 5.2 Install RealSense

Follow the RealSense Jetson guide and prefer the native kernel backend for a
production installation when it is supported by the chosen L4T image. Debian
packages, RSUSB, and native source builds are distinct installation routes.

After installation:

```bash
realsense-viewer
python3 -c "import pyrealsense2 as rs; print(rs.__version__ if hasattr(rs, '__version__') else 'pyrealsense2 ok')"
```

Confirm that the D435i streams color and depth over USB 3 before testing this
repository.

### 5.3 Install the Python runtime

The TensorRT engine session uses CUDA tensors supplied by PyTorch. Install the
NVIDIA PyTorch wheel that matches the installed JetPack, following NVIDIA's
Jetson PyTorch compatibility table. Do not install an arbitrary x86 or generic
CUDA wheel.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-jetson.txt
python -m pip install -e . --no-deps
python - <<'PY'
import cv2, numpy, pyrealsense2, tensorrt, torch, yaml
print('TensorRT', tensorrt.__version__)
print('Torch CUDA', torch.cuda.is_available())
PY
```

`--system-site-packages` exposes the TensorRT, OpenCV, and RealSense bindings
installed for JetPack. If an import fails, fix that system package or matching
JetPack wheel rather than replacing it with an unrelated PyPI build.

### 5.4 Build engines on the target Jetson

```bash
source .venv/bin/activate
python scripts/build_tensorrt_engines.py \
  --trtexec /usr/src/tensorrt/bin/trtexec
ls -lh weights/*.engine
```

The script builds static-batch FP16 engines from the checked-in ONNX files and
fails if `trtexec` does not produce a non-empty file. TensorRT 10 supports the
used `--fp16` build flag. If the platform is later upgraded to TensorRT 11, use
NVIDIA's strongly typed mixed-precision ONNX flow instead of this flag.

Set the engine backend in a Jetson-local config:

```yaml
detection:
  weights: weights/yolov10_sam_robust_standard.engine
  backend: tensorrt
  device: cuda:0
segmentation:
  weights: weights/rgbd_gcnet_l03_robustft.engine
  backend: tensorrt
  device: cuda:0
```

### 5.5 Validate before realtime use

Run in this order:

1. `python -m pytest -q`
2. one fixed offline RGB-D frame with correct recorded intrinsics;
3. the sparse debug viewer with a static scene;
4. a short live D435i run with all recording disabled;
5. a profiling run with only `timings` enabled;
6. a safety-event run only when evidence collection is needed.

Do not use `600/600/320/240` as calibrated intrinsics. They are documentation
examples. Live mode reads the D435i color intrinsics directly.

## 6. Production and diagnostic configurations

Production defaults:

```yaml
profiling: {enabled: false}
debug: {enabled: false}
diagnostics: {enabled: false}
recording:
  enabled: false
```

Example profiling-only override:

```yaml
profiling: {enabled: true}
recording:
  enabled: true
  save_timings: true
  save_detections: false
  save_candidates: false
  save_targets: false
  save_events: false
  save_event_artifacts: false
```

`timings.jsonl` includes model and CPU stage latency, capture time, frame age,
per-queue placeholders, dropped-frame counters, and attachable resource samples.

For Jetson profiling, record the power mode and clocks before comparing runs:

```bash
sudo nvpmodel -q --verbose
sudo jetson_clocks --show
tegrastats --interval 1000 --logfile runs/tegrastats.log
```

Choose an approved power mode for the carrier, cooling, and supply. Use
`jetson_clocks` only for a controlled benchmark if that is part of the test plan.

## 7. Threading acceptance data

Before implementing threads, collect at least:

- capture, detector, tracker/depth, segmentor, grasp, recording, and total time;
- frame age at output and effective FPS;
- queue wait, depth, capacity, and dropped frames for every queue;
- CPU load, process RSS, GPU utilization/memory, temperature, and power;
- target validity rate and each rejection reason.

Start with `capture -> bounded queue -> one serialized GPU worker -> CPU
postprocess/recording`. Queue capacity should initially be 1 or 2 so stale frames
are dropped instead of accumulating latency. Change this only from Orin evidence.

## 8. Safety boundary

The visual runtime fails closed on invalid depth, segmentation, grasp geometry,
static target quality, and same-track temporal jumps. Optional event logs preserve
the rejection evidence. This is a camera-layer safety contract only. Robot-frame
extrinsics, workspace, IK, collision avoidance, emergency stop, actuator limits,
and physical interlocks belong to the robot control system.

## 9. Official platform references

- [NVIDIA JetPack 6.2](https://developer.nvidia.com/embedded/jetpack-sdk-62)
- [NVIDIA JetPack documentation](https://docs.nvidia.com/jetson/jetpack/)
- [NVIDIA PyTorch for Jetson installation](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html)
- [TensorRT Developer Guide](https://docs.nvidia.com/deeplearning/tensorrt/pdf/TensorRT-Developer-Guide.pdf)
- [TensorRT 10 to 11 `trtexec` migration](https://docs.nvidia.com/deeplearning/tensorrt/11.1.0/api/migration/tensorrt-10x-to-11x-trtexec.html)
- [Jetson `tegrastats`](https://docs.nvidia.com/jetson/archives/r34.1/DeveloperGuide/text/AT/JetsonLinuxDevelopmentTools/TegrastatsUtility.html)
- [RealSense Jetson installation](https://github.com/realsenseai/librealsense/blob/master/doc/installation_jetson.md)
