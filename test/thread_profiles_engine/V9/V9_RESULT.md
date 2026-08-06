# V9 多果场景验收结果

**日期**: 2026-08-06
**配置**: `configs/deploy.yaml`（torch-free TensorRT runtime + `reuse_buffers: true`）

## Manifest

| 项目 | 值 |
|---|---|
| status | complete |
| frame_count | 900/900 |
| exit_code | 0 |
| duration | 246.6 s |

## 场景

**每帧 14-16 个苹果**（`confirmed_tracks` P50=14, Max=16, `detector_max_confidence` P50=0.949）

## 900帧核心指标

| 指标 | P50 | P95 | P99 |
|---|---|---|---|
| total_ms | 213.3 | 408.4 | 1662.0 |
| detection_ms | 37.8 | 92.8 | 126.5 |
| depth_fusion_ms | 29.5 | 99.0 | 141.7 |
| segmentation_ms | 24.8 | 98.3 | 431.9 |
| grasp_ms | 68.9 | 189.6 | 1085.8 |
| capture_ms | 33.3 | 134.5 | 215.5 |
| frame_age_ms | 240.7 | 449.3 | 1700.8 |

| 指标 | 值 |
|---|---|
| 有效目标率 | 100% |
| 拒绝率 | 0.17/帧 |
| FPS | 3.74 |

## 分析

- **active_target fast_path** 96.7%：每帧仅 1 个目标进入深度/分割/抓取重管线（`depth_fused_tracks` P50=1.4），即使画面有 14+ 个苹果
- **每 30 帧刷新**（`active_target_refresh` 3.3%）：刷新帧一次性处理全部 ~15 个目标，导致 grasp P99=1085ms、segmentation P99=432ms 尖峰
- 多果场景下快速路径机制有效保持 P50 延迟 213ms，但刷新帧尖峰显著

## V8 vs V9 对比（场景不同，仅供参考）

| 指标 | V8 (无果) | V9 (14+果) |
|---|---|---|
| confirmed_tracks P50 | 0 | 14 |
| total P50 | 17.7 ms | 213.3 ms |
| FPS | 29.35 | 3.74 |
| 有效目标率 | 0% | 100% |

## 环境

**零次 apt / pip / conda 操作。Jetson 环境未修改。**
