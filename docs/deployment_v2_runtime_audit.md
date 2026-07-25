# Deployment 2.0 runtime audit

## Temporal parameters adopted

The deployment tracker now matches the validated dual-threshold experiment:

| Parameter | Value at 30 FPS | Runtime meaning |
|---|---:|---|
| detector confidence | 0.50 | SAM-YOLO deployment acceptance threshold |
| high confidence | 0.50 | may create or continue a track |
| low confidence | 0.10 | may only continue an existing track |
| association IoU | 0.30 | class-consistent IoU match threshold |
| minimum hits | 1 | emit a high-confidence new track immediately |
| maximum lost frames | 30 | retain a track for at most one second |

These values are coupled to the dual-threshold mechanism. They must not be
copied into the former weighted-confidence gate because that does not implement
the evaluated track-creation rule.

## Camera-layer safety line

The camera-side technical line is closed for the current scope:

1. A low-confidence detection cannot create a target track.
2. RGB and depth are aligned in color coordinates and depth remains in mm.
3. Per-track depth history is ROI-remapped and fails closed below the valid-depth ratio.
4. Segmentation, grasp-patch geometry, target score, track confidence, mask quality,
   and surface-normal tilt all have rejection paths.
5. Accepted targets are checked for timestamp regression, pixel jumps, and depth jumps.
6. Rejections can be written to structured event logs with frame/depth/mask/overlay
   evidence when diagnostics are explicitly enabled.
7. All safety, diagnostic, and artifact paths are disabled by default except the
   safety validators themselves.

This closure is limited to producing a camera-frame grasp target. Robot workspace,
inverse kinematics, collision avoidance, emergency stop, actuator limits, and
hardware interlocks remain robot-controller responsibilities and are not claimed
by this repository.

## Profiling and future threading data

`FrameRuntimeTelemetry` is the hand-off contract between future queue workers and
the synchronous pipeline. `timings.jsonl` can contain:

- capture, detector, tracker, depth-fusion, segmentation, grasp, ranking/safety,
  recording, diagnostics, total, and frame-age latency;
- per-queue wait time, current depth, capacity, and dropped-frame counters;
- attachable resource samples such as CPU load, RSS, GPU utilization, GPU memory,
  temperature, and power.

The data schema is complete enough to decide queue boundaries and identify stale
frames. Data collection is intentionally incomplete until the threaded runtime
exists: there are no real queue measurements in the synchronous runner, and Jetson
resource samples must be attached by the future `tegrastats`/device sampler. The
first threading experiment should use bounded queues of size 1 or 2 and keep YOLO
and GCNet GPU execution serialized until Orin measurements show a reason to change it.

## Switches

| Switch | Default | Effect |
|---|---:|---|
| `profiling.enabled` | false | write per-stage and runtime telemetry |
| `debug.enabled` | false | retain `last_debug` masks/detections for the viewer |
| `diagnostics.enabled` | false | enable rejection-event handling |
| `recording.enabled` | false | master switch for run directory and JSONL output |
| `recording.save_*` | false | independently select JSONL/artifact channels |
