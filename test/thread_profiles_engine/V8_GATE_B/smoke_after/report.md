# Thread profiling report

- Frames analyzed: 50 / 60
- Warmup frames excluded: 10
- Target FPS: 30.00
- Effective pipeline FPS: 28.31
- End-to-end FPS including startup: 11.90
- Valid target rate: 0.00%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 33.005 | 33.378 | 38.411 | 40.986 | 42.215 |
| `depth_fusion_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `depth_median_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `depth_remap_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `detection_ms` | 22.874 | 20.367 | 37.726 | 40.650 | 41.257 |
| `diagnostic_ms` | 0.007 | 0.007 | 0.008 | 0.010 | 0.010 |
| `frame_age_ms` | 26.839 | 21.296 | 52.773 | 66.093 | 68.389 |
| `grasp_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `queue_wait_ms.capture_to_inference` | 3.144 | 0.140 | 25.649 | 37.045 | 41.863 |
| `ranking_safety_ms` | 0.028 | 0.025 | 0.032 | 0.065 | 0.077 |
| `recording_ms` | 0.154 | 0.136 | 0.240 | 0.440 | 0.559 |
| `segmentation_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `total_ms` | 23.239 | 20.657 | 38.240 | 41.043 | 41.665 |
| `tracking_ms` | 0.072 | 0.071 | 0.096 | 0.100 | 0.100 |

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
| `workload.detector_max_confidence` | 0.007 | 0.003 | 0.034 | 0.057 |
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
- P95 frame age is 52.8 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.
- Low target validity coincides with sparse detector output; inspect detector maximum confidence, above-threshold candidates, and active-target coast counts before tuning downstream stages.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
