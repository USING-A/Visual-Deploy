# Thread profiling report

- Frames analyzed: 351 / 411
- Warmup frames excluded: 60
- Target FPS: 30.00
- Effective pipeline FPS: 2.40
- End-to-end FPS including startup: 2.28
- Valid target rate: 100.00%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 29.043 | 29.600 | 36.584 | 40.385 | 41.743 |
| `depth_fusion_ms` | 249.679 | 224.217 | 418.201 | 487.563 | 598.571 |
| `detection_ms` | 33.799 | 33.183 | 53.793 | 65.355 | 67.868 |
| `diagnostic_ms` | 0.006 | 0.006 | 0.008 | 0.028 | 0.052 |
| `frame_age_ms` | 428.515 | 398.434 | 677.840 | 790.360 | 912.378 |
| `grasp_ms` | 75.400 | 67.898 | 123.279 | 150.925 | 175.219 |
| `queue_wait_ms.capture_to_inference` | 11.664 | 10.251 | 30.844 | 35.182 | 38.479 |
| `ranking_safety_ms` | 0.573 | 0.469 | 0.844 | 1.147 | 9.481 |
| `recording_ms` | 0.331 | 0.292 | 0.555 | 0.714 | 1.959 |
| `segmentation_ms` | 41.030 | 37.534 | 79.694 | 103.959 | 125.734 |
| `total_ms` | 416.368 | 386.426 | 671.999 | 779.139 | 896.887 |
| `tracking_ms` | 8.874 | 2.947 | 25.721 | 47.044 | 70.339 |

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

- depth_fusion_ms is the dominant stage (60% of median total latency); optimize or isolate this stage first.
- P95 frame age is 677.8 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Dropped frames were observed; compare frame age before increasing any queue capacity.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
