# Thread profiling report

- Frames analyzed: 840 / 900
- Warmup frames excluded: 60
- Target FPS: 30.00
- Effective pipeline FPS: 29.35
- End-to-end FPS including startup: 26.45
- Valid target rate: 0.00%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 33.011 | 32.959 | 39.773 | 42.944 | 48.429 |
| `depth_fusion_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `depth_median_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `depth_remap_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `detection_ms` | 19.276 | 17.481 | 33.485 | 40.370 | 57.901 |
| `diagnostic_ms` | 0.008 | 0.006 | 0.009 | 0.016 | 0.701 |
| `frame_age_ms` | 21.619 | 18.514 | 41.095 | 70.598 | 90.943 |
| `grasp_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `queue_wait_ms.capture_to_inference` | 1.562 | 0.143 | 9.851 | 35.484 | 48.263 |
| `ranking_safety_ms` | 0.028 | 0.027 | 0.036 | 0.057 | 0.211 |
| `recording_ms` | 0.126 | 0.118 | 0.165 | 0.236 | 1.950 |
| `segmentation_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `total_ms` | 19.601 | 17.735 | 33.880 | 40.767 | 58.281 |
| `tracking_ms` | 0.069 | 0.063 | 0.092 | 0.166 | 0.858 |

## Workload summary

| Metric | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| `workload.active_target_coast` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fallback` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_fast_path` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.active_target_refresh` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.coasting_tracks` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.confirmed_tracks` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.deferred_tracks` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.depth_fused_tracks` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.depth_history_frames` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.depth_remap_calls` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.detector_above_threshold_candidates` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.detector_finite_confidence_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
| `workload.detector_max_confidence` | 0.008 | 0.003 | 0.030 | 0.225 |
| `workload.detector_output_candidates` | 300.000 | 300.000 | 300.000 | 300.000 |
| `workload.eligible_detections` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.grasp_candidates` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.grasp_searches` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.pipeline_candidates` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.raw_detections` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.rejections` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.segmentation_calls` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.segmented_tracks` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.tracks` | 0.000 | 0.000 | 0.000 | 0.000 |

## Recommendations

- detection_ms is the dominant stage (98% of median total latency); optimize or isolate this stage first.
- P95 frame age is 41.1 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.
- Low target validity coincides with sparse detector output; inspect detector maximum confidence, above-threshold candidates, and active-target coast counts before tuning downstream stages.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
