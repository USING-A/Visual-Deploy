# Thread profiling report

- Frames analyzed: 50 / 60
- Warmup frames excluded: 10
- Target FPS: 30.00
- Effective pipeline FPS: 29.08
- End-to-end FPS including startup: 9.79
- Valid target rate: 0.00%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 32.525 | 32.962 | 41.772 | 44.418 | 44.592 |
| `depth_fusion_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `depth_median_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `depth_remap_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `detection_ms` | 22.567 | 17.448 | 43.709 | 47.432 | 47.501 |
| `diagnostic_ms` | 0.009 | 0.006 | 0.012 | 0.052 | 0.057 |
| `frame_age_ms` | 28.426 | 23.524 | 58.988 | 80.265 | 83.355 |
| `grasp_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `queue_wait_ms.capture_to_inference` | 4.930 | 0.179 | 20.205 | 35.819 | 36.084 |
| `ranking_safety_ms` | 0.051 | 0.026 | 0.037 | 0.635 | 1.198 |
| `recording_ms` | 0.198 | 0.128 | 0.457 | 1.500 | 1.690 |
| `segmentation_ms` | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| `total_ms` | 22.981 | 17.792 | 44.084 | 47.766 | 47.901 |
| `tracking_ms` | 0.065 | 0.064 | 0.077 | 0.092 | 0.105 |

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
| `workload.detector_max_confidence` | 0.009 | 0.004 | 0.018 | 0.140 |
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
- P95 frame age is 59.0 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.
- Low target validity coincides with sparse detector output; inspect detector maximum confidence, above-threshold candidates, and active-target coast counts before tuning downstream stages.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
