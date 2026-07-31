# Thread profiling report

- Frames analyzed: 840 / 900
- Warmup frames excluded: 60
- Target FPS: 30.00
- Effective pipeline FPS: 8.32
- End-to-end FPS including startup: 7.83
- Valid target rate: 100.00%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 31.991 | 32.405 | 37.122 | 40.024 | 68.108 |
| `depth_fusion_ms` | 21.832 | 22.312 | 36.632 | 61.646 | 97.145 |
| `depth_median_ms` | 15.316 | 13.650 | 29.553 | 36.175 | 60.149 |
| `depth_remap_ms` | 5.035 | 3.067 | 18.786 | 26.291 | 35.035 |
| `detection_ms` | 27.988 | 27.985 | 42.776 | 58.554 | 70.817 |
| `diagnostic_ms` | 0.007 | 0.006 | 0.008 | 0.012 | 0.203 |
| `frame_age_ms` | 129.997 | 117.979 | 203.383 | 245.593 | 337.957 |
| `grasp_ms` | 40.159 | 34.381 | 68.507 | 97.734 | 140.937 |
| `queue_wait_ms.capture_to_inference` | 11.507 | 8.552 | 33.067 | 38.511 | 43.768 |
| `ranking_safety_ms` | 0.394 | 0.347 | 0.591 | 0.884 | 7.571 |
| `recording_ms` | 0.247 | 0.221 | 0.385 | 0.542 | 1.398 |
| `segmentation_ms` | 16.777 | 15.445 | 31.400 | 37.539 | 60.674 |
| `total_ms` | 118.000 | 105.852 | 188.144 | 233.396 | 333.115 |
| `tracking_ms` | 7.069 | 2.514 | 24.363 | 35.147 | 48.283 |

## Workload summary

| Metric | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| `workload.active_target_coast` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fallback` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fast_path` | 0.967 | 1.000 | 1.000 | 1.000 |
| `workload.active_target_refresh` | 0.033 | 0.000 | 0.000 | 1.000 |
| `workload.coasting_tracks` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.confirmed_tracks` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.deferred_tracks` | 0.967 | 1.000 | 1.000 | 1.000 |
| `workload.depth_fused_tracks` | 1.033 | 1.000 | 1.000 | 2.000 |
| `workload.depth_history_frames` | 5.033 | 5.000 | 5.000 | 6.000 |
| `workload.depth_remap_calls` | 3.320 | 4.000 | 4.000 | 4.000 |
| `workload.detector_above_threshold_candidates` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.detector_finite_confidence_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
| `workload.detector_max_confidence` | 0.843 | 0.844 | 0.870 | 0.887 |
| `workload.detector_output_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
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

- P95 frame age is 203.4 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
