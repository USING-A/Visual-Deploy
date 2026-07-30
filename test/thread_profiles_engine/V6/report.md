# Thread profiling report

- Frames analyzed: 840 / 900
- Warmup frames excluded: 60
- Target FPS: 30.00
- Effective pipeline FPS: 8.29
- End-to-end FPS including startup: 7.89
- Valid target rate: 100.00%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 31.870 | 32.457 | 37.293 | 41.076 | 44.790 |
| `depth_fusion_ms` | 16.441 | 13.891 | 34.066 | 52.727 | 64.821 |
| `depth_median_ms` | 13.460 | 9.526 | 30.264 | 35.604 | 61.015 |
| `depth_remap_ms` | 1.385 | 0.311 | 5.276 | 20.586 | 26.455 |
| `detection_ms` | 29.676 | 28.722 | 47.079 | 60.983 | 71.998 |
| `diagnostic_ms` | 0.006 | 0.006 | 0.008 | 0.009 | 0.265 |
| `frame_age_ms` | 130.453 | 117.653 | 208.734 | 272.562 | 339.674 |
| `grasp_ms` | 43.474 | 36.505 | 79.129 | 100.384 | 156.967 |
| `queue_wait_ms.capture_to_inference` | 11.357 | 7.979 | 32.453 | 36.675 | 42.965 |
| `ranking_safety_ms` | 0.402 | 0.361 | 0.590 | 0.837 | 10.352 |
| `recording_ms` | 0.261 | 0.237 | 0.393 | 0.562 | 2.000 |
| `segmentation_ms` | 17.427 | 16.019 | 32.848 | 38.254 | 113.263 |
| `total_ms` | 118.582 | 105.805 | 194.992 | 264.003 | 327.289 |
| `tracking_ms` | 6.903 | 2.466 | 22.630 | 45.701 | 48.793 |

## Workload summary

| Metric | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| `workload.active_target_coast` | 0.037 | 0.000 | 0.000 | 1.000 |
| `workload.active_target_fallback` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fast_path` | 0.882 | 1.000 | 1.000 | 1.000 |
| `workload.active_target_refresh` | 0.032 | 0.000 | 0.000 | 1.000 |
| `workload.coasting_tracks` | 0.037 | 0.000 | 0.000 | 1.000 |
| `workload.confirmed_tracks` | 1.607 | 2.000 | 2.000 | 2.000 |
| `workload.deferred_tracks` | 0.619 | 1.000 | 1.000 | 1.000 |
| `workload.depth_fused_tracks` | 1.025 | 1.000 | 1.000 | 2.000 |
| `workload.depth_history_frames` | 4.249 | 5.000 | 5.000 | 6.000 |
| `workload.depth_remap_calls` | 0.496 | 0.000 | 3.000 | 4.000 |
| `workload.detector_above_threshold_candidates` | 1.607 | 2.000 | 2.000 | 2.000 |
| `workload.detector_finite_confidence_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
| `workload.detector_max_confidence` | 0.592 | 0.596 | 0.663 | 0.757 |
| `workload.detector_output_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
| `workload.eligible_detections` | 1.607 | 2.000 | 2.000 | 2.000 |
| `workload.grasp_candidates` | 1.025 | 1.000 | 1.000 | 2.000 |
| `workload.grasp_searches` | 1.025 | 1.000 | 1.000 | 2.000 |
| `workload.pipeline_candidates` | 1.025 | 1.000 | 1.000 | 2.000 |
| `workload.raw_detections` | 1.607 | 2.000 | 2.000 | 2.000 |
| `workload.rejections` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.segmentation_calls` | 1.025 | 1.000 | 1.000 | 2.000 |
| `workload.segmented_tracks` | 1.025 | 1.000 | 1.000 | 2.000 |
| `workload.tracks` | 1.644 | 2.000 | 2.000 | 2.000 |

## Recommendations

- grasp_ms is the dominant stage (37% of median total latency); optimize or isolate this stage first.
- P95 frame age is 208.7 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.
- Jetson GPU utilization is saturated at P95; keep YOLO and GCNet serialized before testing GPU concurrency.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
