# Thread profiling report

- Frames analyzed: 840 / 900
- Warmup frames excluded: 60
- Target FPS: 30.00
- Effective pipeline FPS: 1.76
- End-to-end FPS including startup: 1.76
- Valid target rate: 100.00%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 30.322 | 31.199 | 36.559 | 40.953 | 52.165 |
| `depth_fusion_ms` | 351.862 | 299.685 | 632.109 | 657.842 | 690.257 |
| `detection_ms` | 37.142 | 35.498 | 64.428 | 78.700 | 93.503 |
| `diagnostic_ms` | 0.006 | 0.006 | 0.008 | 0.009 | 0.019 |
| `frame_age_ms` | 576.883 | 499.281 | 1000.704 | 1032.007 | 1103.488 |
| `grasp_ms` | 103.972 | 89.932 | 190.843 | 208.453 | 218.537 |
| `queue_wait_ms.capture_to_inference` | 10.819 | 6.771 | 31.661 | 37.873 | 42.825 |
| `ranking_safety_ms` | 0.588 | 0.534 | 0.819 | 1.055 | 14.674 |
| `recording_ms` | 0.372 | 0.332 | 0.573 | 0.716 | 1.124 |
| `segmentation_ms` | 48.623 | 41.733 | 93.058 | 123.124 | 132.001 |
| `total_ms` | 565.533 | 487.379 | 991.314 | 1017.853 | 1067.797 |
| `tracking_ms` | 12.440 | 3.910 | 41.830 | 48.437 | 68.297 |

## Workload summary

| Metric | Mean | P50 | P95 | Max |
|---|---:|---:|---:|---:|
| `workload.confirmed_tracks` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.depth_fused_tracks` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.eligible_detections` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.grasp_candidates` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.grasp_searches` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.pipeline_candidates` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.raw_detections` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.rejections` | 0.000 | 0.000 | 0.000 | 0.000 |
| `workload.segmentation_calls` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.segmented_tracks` | 2.000 | 2.000 | 2.000 | 2.000 |
| `workload.tracks` | 2.000 | 2.000 | 2.000 | 2.000 |

## Recommendations

- depth_fusion_ms is the dominant stage (62% of median total latency); optimize or isolate this stage first.
- P95 frame age is 1000.7 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
