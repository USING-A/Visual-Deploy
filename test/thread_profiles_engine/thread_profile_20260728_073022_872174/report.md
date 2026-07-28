# Thread profiling report

- Frames analyzed: 280 / 300
- Warmup frames excluded: 20
- Target FPS: 30.00
- Effective pipeline FPS: 14.36
- End-to-end FPS including startup: 10.31
- Valid target rate: 22.50%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 11.416 | 10.769 | 17.014 | 18.773 | 20.107 |
| `depth_fusion_ms` | 24.545 | 0.000 | 77.446 | 81.865 | 85.255 |
| `detection_ms` | 22.228 | 22.193 | 26.919 | 31.283 | 33.675 |
| `diagnostic_ms` | 0.006 | 0.006 | 0.008 | 0.009 | 0.015 |
| `frame_age_ms` | 57.419 | 24.571 | 133.503 | 141.557 | 145.978 |
| `grasp_ms` | 5.911 | 0.000 | 18.591 | 20.339 | 22.433 |
| `ranking_safety_ms` | 0.129 | 0.023 | 0.375 | 0.421 | 0.709 |
| `recording_ms` | 0.153 | 0.133 | 0.234 | 0.332 | 0.445 |
| `segmentation_ms` | 3.111 | 0.000 | 11.970 | 13.282 | 16.506 |
| `total_ms` | 57.169 | 24.284 | 133.204 | 141.303 | 145.739 |
| `tracking_ms` | 0.469 | 0.080 | 1.385 | 1.650 | 2.060 |

## Recommendations

- depth_fusion_ms is the dominant stage (43% of median total latency); optimize or isolate this stage first.
- P95 frame age is 133.5 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- No real queue metrics were recorded; implement bounded queues before using this run to choose queue capacities or worker counts.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
