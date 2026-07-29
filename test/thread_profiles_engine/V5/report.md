# Thread profiling report

- Frames analyzed: 840 / 900
- Warmup frames excluded: 60
- Target FPS: 30.00
- Effective pipeline FPS: 10.33
- End-to-end FPS including startup: 9.56
- Valid target rate: 76.79%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 32.158 | 32.518 | 38.290 | 43.686 | 52.009 |
| `depth_fusion_ms` | 15.057 | 12.447 | 34.843 | 61.110 | 98.122 |
| `depth_median_ms` | 10.057 | 7.313 | 27.767 | 35.099 | 66.052 |
| `depth_remap_ms` | 3.693 | 2.465 | 17.611 | 26.616 | 30.850 |
| `detection_ms` | 28.129 | 28.379 | 44.335 | 60.946 | 92.430 |
| `diagnostic_ms` | 0.007 | 0.006 | 0.008 | 0.022 | 0.324 |
| `frame_age_ms` | 104.243 | 107.205 | 179.165 | 250.353 | 308.145 |
| `grasp_ms` | 29.611 | 31.133 | 65.396 | 98.773 | 128.939 |
| `queue_wait_ms.capture_to_inference` | 10.308 | 6.490 | 32.725 | 38.925 | 53.484 |
| `ranking_safety_ms` | 0.338 | 0.328 | 0.558 | 1.101 | 12.168 |
| `recording_ms` | 0.223 | 0.210 | 0.383 | 0.541 | 0.949 |
| `segmentation_ms` | 12.930 | 10.538 | 30.608 | 36.931 | 54.991 |
| `total_ms` | 93.445 | 96.476 | 168.801 | 239.034 | 301.272 |
| `tracking_ms` | 4.332 | 1.629 | 18.244 | 26.794 | 50.499 |

## Workload summary

| Metric | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| `workload.active_target_coast` | 0.065 | 0.000 | 1.000 | 1.000 |
| `workload.active_target_fallback` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fast_path` | 0.699 | 1.000 | 1.000 | 1.000 |
| `workload.active_target_refresh` | 0.023 | 0.000 | 0.000 | 1.000 |
| `workload.coasting_tracks` | 0.065 | 0.000 | 1.000 | 1.000 |
| `workload.confirmed_tracks` | 1.382 | 1.000 | 3.000 | 5.000 |
| `workload.deferred_tracks` | 0.646 | 0.000 | 2.000 | 4.000 |
| `workload.depth_fused_tracks` | 0.796 | 1.000 | 1.000 | 3.000 |
| `workload.depth_history_frames` | 3.405 | 5.000 | 5.000 | 7.000 |
| `workload.depth_remap_calls` | 2.239 | 3.000 | 4.000 | 4.000 |
| `workload.detector_above_threshold_candidates` | 1.382 | 1.000 | 3.000 | 5.000 |
| `workload.detector_finite_confidence_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
| `workload.detector_max_confidence` | 0.594 | 0.600 | 0.799 | 0.856 |
| `workload.detector_output_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
| `workload.eligible_detections` | 1.382 | 1.000 | 3.000 | 5.000 |
| `workload.grasp_candidates` | 0.796 | 1.000 | 1.000 | 3.000 |
| `workload.grasp_searches` | 0.796 | 1.000 | 1.000 | 3.000 |
| `workload.pipeline_candidates` | 0.796 | 1.000 | 1.000 | 3.000 |
| `workload.raw_detections` | 1.382 | 1.000 | 3.000 | 5.000 |
| `workload.rejections` | 0.013 | 0.000 | 0.000 | 1.000 |
| `workload.segmentation_calls` | 0.796 | 1.000 | 1.000 | 3.000 |
| `workload.segmented_tracks` | 0.796 | 1.000 | 1.000 | 3.000 |
| `workload.tracks` | 1.448 | 1.000 | 3.000 | 5.000 |

## Recommendations

- P95 frame age is 179.2 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.
- Jetson GPU utilization is saturated at P95; keep YOLO and GCNet serialized before testing GPU concurrency.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
