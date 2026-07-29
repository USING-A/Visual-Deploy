# Thread profiling report

- Frames analyzed: 840 / 900
- Warmup frames excluded: 60
- Target FPS: 30.00
- Effective pipeline FPS: 6.21
- End-to-end FPS including startup: 5.82
- Valid target rate: 75.83%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 53.330 | 37.338 | 158.873 | 233.290 | 318.219 |
| `depth_fusion_ms` | 20.868 | 11.024 | 82.196 | 138.498 | 247.419 |
| `depth_median_ms` | 12.795 | 6.373 | 49.290 | 110.534 | 152.670 |
| `depth_remap_ms` | 5.242 | 1.181 | 30.700 | 65.081 | 200.520 |
| `detection_ms` | 50.174 | 41.656 | 109.282 | 165.815 | 228.317 |
| `diagnostic_ms` | 0.016 | 0.005 | 0.008 | 0.041 | 7.565 |
| `frame_age_ms` | 175.639 | 166.636 | 350.260 | 431.779 | 537.850 |
| `grasp_ms` | 44.257 | 36.878 | 124.644 | 170.967 | 323.053 |
| `queue_wait_ms.capture_to_inference` | 25.314 | 18.894 | 75.410 | 113.077 | 188.572 |
| `ranking_safety_ms` | 0.487 | 0.338 | 0.827 | 4.382 | 32.191 |
| `recording_ms` | 0.427 | 0.207 | 0.566 | 4.462 | 47.258 |
| `segmentation_ms` | 25.448 | 16.922 | 79.879 | 131.430 | 178.021 |
| `total_ms` | 149.511 | 144.249 | 315.426 | 399.728 | 468.843 |
| `tracking_ms` | 3.360 | 1.064 | 18.237 | 45.095 | 167.497 |

## Workload summary

| Metric | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| `workload.active_target_fallback` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fast_path` | 0.580 | 1.000 | 1.000 | 1.000 |
| `workload.active_target_refresh` | 0.020 | 0.000 | 0.000 | 1.000 |
| `workload.confirmed_tracks` | 0.764 | 1.000 | 1.000 | 2.000 |
| `workload.deferred_tracks` | 0.005 | 0.000 | 0.000 | 1.000 |
| `workload.depth_fused_tracks` | 0.760 | 1.000 | 1.000 | 2.000 |
| `workload.depth_history_frames` | 2.630 | 2.000 | 5.000 | 5.000 |
| `workload.depth_remap_calls` | 1.585 | 1.000 | 4.000 | 4.000 |
| `workload.eligible_detections` | 0.764 | 1.000 | 1.000 | 2.000 |
| `workload.grasp_candidates` | 0.760 | 1.000 | 1.000 | 2.000 |
| `workload.grasp_searches` | 0.760 | 1.000 | 1.000 | 2.000 |
| `workload.pipeline_candidates` | 0.760 | 1.000 | 1.000 | 2.000 |
| `workload.raw_detections` | 0.764 | 1.000 | 1.000 | 2.000 |
| `workload.rejections` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.segmentation_calls` | 0.760 | 1.000 | 1.000 | 2.000 |
| `workload.segmented_tracks` | 0.760 | 1.000 | 1.000 | 2.000 |
| `workload.tracks` | 0.764 | 1.000 | 1.000 | 2.000 |

## Recommendations

- P95 frame age is 350.3 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.
- Jetson GPU utilization is saturated at P95; keep YOLO and GCNet serialized before testing GPU concurrency.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
