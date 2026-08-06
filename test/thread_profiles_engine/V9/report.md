# Thread profiling report

- Frames analyzed: 840 / 900
- Warmup frames excluded: 60
- Target FPS: 30.00
- Effective pipeline FPS: 3.74
- End-to-end FPS including startup: 3.65
- Valid target rate: 100.00%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 45.971 | 33.299 | 134.470 | 215.527 | 313.653 |
| `depth_fusion_ms` | 38.145 | 29.486 | 98.978 | 141.694 | 273.121 |
| `depth_median_ms` | 26.060 | 22.160 | 75.917 | 102.901 | 164.712 |
| `depth_remap_ms` | 6.825 | 1.665 | 35.654 | 66.141 | 105.804 |
| `detection_ms` | 45.298 | 37.832 | 92.842 | 126.476 | 170.566 |
| `diagnostic_ms` | 0.006 | 0.005 | 0.008 | 0.026 | 0.162 |
| `frame_age_ms` | 288.563 | 240.671 | 449.328 | 1700.849 | 1934.467 |
| `grasp_ms` | 106.239 | 68.901 | 189.618 | 1085.800 | 1245.292 |
| `queue_wait_ms.capture_to_inference` | 27.123 | 24.221 | 70.469 | 91.480 | 136.194 |
| `ranking_safety_ms` | 0.924 | 0.338 | 1.505 | 18.744 | 49.425 |
| `recording_ms` | 0.356 | 0.199 | 0.905 | 2.487 | 16.388 |
| `segmentation_ms` | 41.902 | 24.794 | 98.345 | 431.927 | 521.365 |
| `total_ms` | 260.744 | 213.325 | 408.357 | 1661.978 | 1928.662 |
| `tracking_ms` | 21.385 | 7.994 | 67.929 | 97.973 | 150.187 |

## Workload summary

| Metric | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| `workload.active_target_coast` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fallback` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fast_path` | 0.967 | 1.000 | 1.000 | 1.000 |
| `workload.active_target_refresh` | 0.033 | 0.000 | 0.000 | 1.000 |
| `workload.coasting_tracks` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.confirmed_tracks` | 14.173 | 14.000 | 15.000 | 16.000 |
| `workload.deferred_tracks` | 12.740 | 13.000 | 14.000 | 15.000 |
| `workload.depth_fused_tracks` | 1.432 | 1.000 | 1.000 | 15.000 |
| `workload.depth_history_frames` | 5.318 | 5.000 | 5.000 | 19.000 |
| `workload.depth_remap_calls` | 1.564 | 1.000 | 4.000 | 4.000 |
| `workload.detector_above_threshold_candidates` | 14.173 | 14.000 | 15.000 | 16.000 |
| `workload.detector_finite_confidence_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
| `workload.detector_max_confidence` | 0.950 | 0.949 | 0.957 | 0.964 |
| `workload.detector_output_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
| `workload.eligible_detections` | 14.173 | 14.000 | 15.000 | 16.000 |
| `workload.grasp_candidates` | 1.432 | 1.000 | 1.000 | 15.000 |
| `workload.grasp_searches` | 1.432 | 1.000 | 1.000 | 15.000 |
| `workload.pipeline_candidates` | 1.432 | 1.000 | 1.000 | 15.000 |
| `workload.raw_detections` | 14.173 | 14.000 | 15.000 | 16.000 |
| `workload.rejections` | 0.167 | 0.000 | 0.000 | 6.000 |
| `workload.segmentation_calls` | 1.432 | 1.000 | 1.000 | 15.000 |
| `workload.segmented_tracks` | 1.432 | 1.000 | 1.000 | 15.000 |
| `workload.tracks` | 14.173 | 14.000 | 15.000 | 16.000 |

## Recommendations

- grasp_ms is the dominant stage (41% of median total latency); optimize or isolate this stage first.
- P95 frame age is 449.3 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.
- Jetson GPU utilization is saturated at P95; keep YOLO and GCNet serialized before testing GPU concurrency.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
