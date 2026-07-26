# Visual-Deploy 2.0 release confidence audit

Date: 2026-07-26

## Decision

| Scope | Decision | Confidence |
|---|---|---:|
| Repository, ONNX models, synchronous pipeline, safety gates | Go | 97% |
| Automated profile collection and analysis | Go | 96% |
| Jetson Orin NX plus D435i plus device-built TensorRT engines | Conditional Go | below 95% until the device checklist passes |

The remaining Jetson uncertainty is hardware/runtime evidence, not a known open
code defect. Do not relabel the full device deployment as 95% accepted until the
commands in the final section pass on the target Orin NX.

## Verified release evidence

| Area | Evidence | Result |
|---|---|---|
| Detector artifact | SHA-256 `3632ee3f270ddcc3d849cb143f0fe9fca7fa6f220b04ae7b202b54bf168a5330` | matches metadata |
| Segmentor artifact | SHA-256 `a90917388178c223c441e66bd0c363c7c2af8e948ddf9cbaa92bf763d4390330` | matches metadata |
| Detector contract | `images [1,3,640,640] -> output0 [1,300,6]` | loaded and inferred |
| Segmentor contract | BGR/depth `[1,3/1,256,256] -> prob [1,1,256,256]` | loaded, finite output |
| Standard YOLOv10 | export metadata reports no ECA and removed training-only SAM distiller | pass |
| Temporal parameters | high 0.50, low 0.10, IoU 0.30, min hits 1, max lost 30 | config and tests match |
| CPU/production parity | tracking, depth fusion, safety, profiling/debug/diagnostics | exact match |
| Safety default | static and continuity gates enabled in both deploy configs | pass |
| Debug/recording default | profiling, debug, diagnostics, recording all disabled | pass |
| Python compatibility | compile under Python 3.9 and 3.10 | pass |
| Test suite | `196 passed` | pass |

## Real-model execution

The final paired profile used one 1280x720 apple image, synthetic aligned flat
depth, four YOLO detections, both real ONNX models, production safety gates, 12
frames, and two warmup frames.

| Metric | Original audited code | Final code |
|---|---:|---:|
| Mean total latency | 1785.6 ms | 361.1 ms |
| Mean grasp latency | 1668.0 ms | 247.7 ms |
| Effective throughput | 0.56 FPS | 2.75 FPS |
| Valid target rate | 100% | 100% |

The selected track, pixel, depth, normal, approach axis, and camera-frame point
were unchanged. Five distinct apple images with two to five detections also ran
through detector, segmentor, geometry, ranking, and safety without exceptions.

This is a functional acceptance result, not a 30 FPS claim. The remaining grasp
stage is candidate-count dependent and should be the first target of later worker
and algorithm experiments.

## Automated profiler evidence

| Probe | Expected status | Observed |
|---|---|---|
| Real ONNX 12-frame run | `complete`, exit 0 | pass |
| Forced timeout | `timed_out`, exit 124 | pass |
| Missing RGB child failure | `child_failed`, nonzero | pass |
| Early resource sampler exit | `incomplete_telemetry`, exit 3 | pass |
| Orphan process check | no matching Python process | pass |

All probes preserved `manifest.json`, logs, resource samples, and analysis output.
No incomplete probe was reported as complete.

## TensorRT and device protection

- TensorRT 8 and 10 loader paths check shape binding, tensor address binding,
  unresolved output shape, execution return status, CUDA availability, and dtype.
- Engine builds use a temporary sibling and atomically replace the final engine
  only after `trtexec` succeeds and produces a non-empty file.
- A failed engine build preserves the previous engine and removes its partial
  temporary output.
- Engines must still be built and loaded on the actual target JetPack/device.

## Jetson acceptance gate

Run on the final Orin NX, with the final carrier, cooling, power mode, D435i,
JetPack, and TensorRT engines:

```bash
python scripts/build_tensorrt_engines.py \
  --trtexec /usr/src/tensorrt/bin/trtexec

python scripts/collect_thread_profile.py \
  --mode realtime \
  --config configs/deploy.jetson.local.yaml \
  --frames 900 \
  --warmup-frames 60 \
  --sample-interval-ms 1000 \
  --timeout-s 180
```

Promote the full device deployment to 95% Go only when all conditions hold:

1. both engines build on the target and load through the TensorRT backend;
2. the collector returns `complete` and exit code 0 with exactly 900 timing rows;
3. tegrastats samples contain GPU utilization, temperatures, RAM, and input power;
4. D435i capture stays aligned and no camera/child errors appear;
5. P95 latency, frame age, temperature, power, and valid-target rate meet the
   application acceptance thresholds defined for the robot test;
6. no orphan process, unbounded queue growth, or repeated safety rejection is present.
