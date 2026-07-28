# Thread profiling report

- Frames analyzed: 280 / 300
- Warmup frames excluded: 20
- Target FPS: 30.00
- Effective pipeline FPS: 2.03
- End-to-end FPS including startup: 2.01
- Valid target rate: 50.71%

## Timing summary

| Metric | Mean | P50 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|
| `capture_ms` | 10.237 | 9.659 | 13.623 | 17.543 | 30.377 |
| `depth_fusion_ms` | 87.874 | 87.127 | 99.076 | 116.320 | 149.393 |
| `detection_ms` | 236.236 | 220.382 | 349.742 | 431.011 | 491.256 |
| `diagnostic_ms` | 0.007 | 0.007 | 0.008 | 0.009 | 0.011 |
| `frame_age_ms` | 480.391 | 456.164 | 651.610 | 778.985 | 857.944 |
| `grasp_ms` | 19.128 | 18.590 | 24.905 | 29.008 | 30.328 |
| `ranking_safety_ms` | 0.309 | 0.289 | 0.380 | 0.691 | 2.073 |
| `recording_ms` | 0.224 | 0.191 | 0.248 | 0.780 | 4.822 |
| `segmentation_ms` | 133.465 | 123.406 | 203.407 | 305.857 | 380.245 |
| `total_ms` | 480.154 | 455.946 | 651.389 | 778.756 | 857.711 |
| `tracking_ms` | 1.074 | 0.898 | 2.012 | 6.055 | 7.510 |

## Recommendations

- detection_ms is the dominant stage (49% of median total latency); optimize or isolate this stage first.
- P95 frame age is 651.6 ms, above the 33.3 ms frame budget; use a bounded latest-frame queue.
- Jetson CPU utilization is high at P95; move recording and non-stateful postprocess work off the inference path.
- No real queue metrics were recorded; implement bounded queues before using this run to choose queue capacities or worker counts.

Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.
