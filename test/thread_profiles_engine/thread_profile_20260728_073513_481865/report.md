# Thread profiling report

- Frames analyzed: 280 / 300
- Warmup frames excluded: 20
- Target FPS: 30.00
- Effective pipeline FPS: 9.83
- End-to-end FPS including startup: 7.90
- Valid target rate: 97.14%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 16.056 | 15.701 | 19.921 | 21.293 | 23.578 |
| `depth_fusion_ms` | 29.742 | 30.337 | 35.734 | 39.621 | 62.495 |
| `detection_ms` | 22.953 | 23.450 | 29.505 | 31.349 | 32.882 |
| `diagnostic_ms` | 0.006 | 0.005 | 0.007 | 0.010 | 0.059 |
| `frame_age_ms` | 84.960 | 85.337 | 97.113 | 106.845 | 141.874 |
| `grasp_ms` | 18.298 | 18.405 | 21.790 | 25.098 | 35.262 |
| `ranking_safety_ms` | 0.312 | 0.308 | 0.381 | 0.494 | 0.743 |
| `recording_ms` | 0.205 | 0.198 | 0.249 | 0.363 | 0.600 |
| `segmentation_ms` | 10.952 | 10.700 | 16.452 | 18.791 | 27.546 |
| `total_ms` | 84.711 | 85.094 | 96.736 | 106.605 | 141.641 |
| `tracking_ms` | 0.872 | 0.855 | 1.108 | 1.341 | 1.734 |

## Recommendations

- depth_fusion_ms is the dominant stage (35% of median total latency); optimize or isolate this stage first.
- P95 frame age is 97.1 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Jetson GPU utilization is saturated at P95; keep YOLO and GCNet serialized before testing GPU concurrency.
- No real queue metrics were recorded; implement bounded queues before using this run to choose queue capacities or worker counts.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
