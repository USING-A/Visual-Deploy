# Thread profiling report

- Frames analyzed: 840 / 900
- Warmup frames excluded: 60
- Target FPS: 30.00
- Effective pipeline FPS: 4.98
- End-to-end FPS including startup: 4.67
- Valid target rate: 100.00%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 55.339 | 35.558 | 176.616 | 260.588 | 543.605 |
| `depth_fusion_ms` | 33.507 | 24.072 | 90.533 | 142.904 | 201.733 |
| `depth_median_ms` | 21.732 | 12.651 | 65.959 | 115.700 | 198.247 |
| `depth_remap_ms` | 8.947 | 3.555 | 34.425 | 76.569 | 127.320 |
| `detection_ms` | 51.998 | 41.648 | 122.639 | 162.314 | 230.926 |
| `diagnostic_ms` | 0.007 | 0.006 | 0.008 | 0.044 | 0.599 |
| `frame_age_ms` | 221.497 | 209.305 | 375.293 | 475.547 | 630.643 |
| `grasp_ms` | 61.648 | 48.560 | 144.598 | 204.826 | 307.362 |
| `queue_wait_ms.capture_to_inference` | 28.688 | 20.760 | 82.895 | 116.669 | 190.193 |
| `ranking_safety_ms` | 0.560 | 0.369 | 1.318 | 4.110 | 18.856 |
| `recording_ms` | 0.424 | 0.222 | 0.741 | 3.506 | 29.722 |
| `segmentation_ms` | 32.117 | 23.693 | 82.266 | 112.335 | 201.988 |
| `total_ms` | 191.837 | 179.219 | 337.302 | 434.801 | 528.510 |
| `tracking_ms` | 6.216 | 1.892 | 28.154 | 62.551 | 141.276 |

## Workload summary

| Metric | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| `workload.active_target_coast` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fallback` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fast_path` | 0.962 | 1.000 | 1.000 | 1.000 |
| `workload.active_target_refresh` | 0.033 | 0.000 | 0.000 | 1.000 |
| `workload.coasting_tracks` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.confirmed_tracks` | 1.931 | 2.000 | 2.000 | 3.000 |
| `workload.deferred_tracks` | 0.901 | 1.000 | 1.000 | 2.000 |
| `workload.depth_fused_tracks` | 1.030 | 1.000 | 1.000 | 2.000 |
| `workload.depth_history_frames` | 4.983 | 5.000 | 5.000 | 6.000 |
| `workload.depth_remap_calls` | 3.311 | 4.000 | 4.000 | 4.000 |
| `workload.detector_above_threshold_candidates` | 1.931 | 2.000 | 2.000 | 3.000 |
| `workload.detector_finite_confidence_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
| `workload.detector_max_confidence` | 0.665 | 0.667 | 0.737 | 0.935 |
| `workload.detector_output_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
| `workload.eligible_detections` | 1.931 | 2.000 | 2.000 | 3.000 |
| `workload.grasp_candidates` | 1.030 | 1.000 | 1.000 | 2.000 |
| `workload.grasp_searches` | 1.030 | 1.000 | 1.000 | 2.000 |
| `workload.pipeline_candidates` | 1.030 | 1.000 | 1.000 | 2.000 |
| `workload.raw_detections` | 1.931 | 2.000 | 2.000 | 3.000 |
| `workload.rejections` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.segmentation_calls` | 1.030 | 1.000 | 1.000 | 2.000 |
| `workload.segmented_tracks` | 1.030 | 1.000 | 1.000 | 2.000 |
| `workload.tracks` | 1.931 | 2.000 | 2.000 | 3.000 |

## Recommendations

- P95 frame age is 375.3 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.
- Jetson GPU utilization is saturated at P95; keep YOLO and GCNet serialized before testing GPU concurrency.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
