# Thread profiling report

- Frames analyzed: 840 / 900
- Warmup frames excluded: 60
- Target FPS: 30.00
- Effective pipeline FPS: 4.11
- End-to-end FPS including startup: 3.94
- Valid target rate: 100.00%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 29.682 | 30.350 | 36.445 | 39.791 | 44.652 |
| `depth_fusion_ms` | 126.013 | 108.187 | 226.938 | 271.283 | 319.196 |
| `detection_ms` | 33.536 | 32.669 | 51.545 | 69.224 | 96.237 |
| `diagnostic_ms` | 0.007 | 0.006 | 0.008 | 0.016 | 0.225 |
| `frame_age_ms` | 253.100 | 236.855 | 402.698 | 471.671 | 540.437 |
| `grasp_ms` | 44.048 | 36.710 | 82.244 | 117.971 | 154.256 |
| `queue_wait_ms.capture_to_inference` | 11.591 | 9.004 | 33.084 | 37.012 | 45.478 |
| `ranking_safety_ms` | 0.436 | 0.374 | 0.649 | 0.784 | 9.833 |
| `recording_ms` | 0.276 | 0.244 | 0.426 | 0.658 | 3.321 |
| `segmentation_ms` | 22.224 | 21.259 | 50.636 | 77.550 | 96.077 |
| `total_ms` | 240.997 | 223.927 | 391.811 | 453.801 | 536.027 |
| `tracking_ms` | 8.922 | 3.047 | 25.343 | 47.194 | 70.445 |

## Workload summary

| Metric | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| `workload.active_target_fallback` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fast_path` | 0.967 | 1.000 | 1.000 | 1.000 |
| `workload.active_target_refresh` | 0.033 | 0.000 | 0.000 | 1.000 |
| `workload.confirmed_tracks` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.deferred_tracks` | 0.967 | 1.000 | 1.000 | 1.000 |
| `workload.depth_fused_tracks` | 1.033 | 1.000 | 1.000 | 2.000 |
| `workload.eligible_detections` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.grasp_candidates` | 1.033 | 1.000 | 1.000 | 2.000 |
| `workload.grasp_searches` | 1.033 | 1.000 | 1.000 | 2.000 |
| `workload.pipeline_candidates` | 1.033 | 1.000 | 1.000 | 2.000 |
| `workload.raw_detections` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.rejections` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.segmentation_calls` | 1.033 | 1.000 | 1.000 | 2.000 |
| `workload.segmented_tracks` | 1.033 | 1.000 | 1.000 | 2.000 |
| `workload.tracks` | 2.000 | 2.000 | 2.000 | 2.000 |

## Recommendations

- depth_fusion_ms is the dominant stage (52% of median total latency); optimize or isolate this stage first.
- P95 frame age is 402.7 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
