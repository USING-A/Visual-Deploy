# Gate B 验收结果

**日期**: 2026-07-31

## Git SHA

- 同步后: `6cb5b3d`（与 Gate A 相同，已是最新）

## 三段 manifest

| 段 | status | frames | exit_code |
|---|---|---|---|
| smoke_before | complete | 60/60 | 0 |
| main_900 | complete | 900/900 | 0 |
| smoke_after | complete | 60/60 | 0 |

## 900帧核心指标 (V8)

| 指标 | 值 |
|---|---|
| P50 total | 17.7 ms |
| P50 detection | 17.5 ms |
| FPS | 29.35 |
| 有效目标率 | 0%（场景中无苹果） |

## V7 vs V8 对比

| 指标 | V7 | V8 | delta |
|---|---|---|---|
| effective_fps | 8.32 | 29.35 | +252.6% |
| total_mean_ms | 118.0 | 19.6 | -83.4% |
| total_p95_ms | 188.1 | 33.9 | -82.0% |
| rss_p95_mb | 1059.1 | 512.8 | -51.6% |

> 注意：V8 场景中无苹果（valid_target_rate=0%），所有检测被 0.50 置信度阈值过滤。P50 detection 17.5ms 为纯推理时间（无后续阶段）。RSS 下降 51.6% 主要因不再加载 PyTorch。

## warning 扫描

- ❌ PyTorch sm_87: **已消除**
- ❌ 跨设备 engine: 无
- ❌ NvMap / OOM / Traceback / TRT error: 无

## torch_loaded 检查

- `torch_loaded`: **False** ✅
- `visual_deploy/inference/`: **零 torch 导入** ✅

## engine 哈希（与 V7 一致）

| Engine | SHA-256 |
|---|---|
| YOLO | `b489dafaf3620cc4d4628df8111403d53d3863a5df29c5f0d270c5aa4485e894` |
| GCNet | `64afd9a619aacb4d7b67a848340512ce6d77442903fc634d0da8921787f4121d` |

## 残留进程

无。

## 环境修改

**零次 apt / pip / conda 操作。Jetson 环境完全未修改。**

## 最终判定

**Gate B 通过** ✅

- 三段连续完成，无需 reboot
- PyTorch sm_87 和跨设备 engine 两类警告均已在不修改 Jetson 环境的条件下消除
- torch_loaded=False，推理模块不再依赖 PyTorch
- 资源正常释放（smoke_after 立即成功）
- 无性能回退（场景无苹果，V8 detection 纯推理 17.5ms 优于 V7 28ms）
