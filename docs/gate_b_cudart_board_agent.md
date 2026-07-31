# Gate B：torch-free CUDA Runtime 板端验收

## 1. 目标和完成边界

Gate B代码侧已经完成以下改造：

- TensorRT运行时不再导入或调用PyTorch；
- 直接加载JetPack现有的`libcudart`；
- 检测与分割session分别持有一条非阻塞CUDA stream；
- H2D、TensorRT执行、D2H按顺序进入同一stream，每帧只同步一次；
- 固定shape时复用设备缓冲区，shape或dtype变化时安全重建；
- 执行失败、关闭窗口和正常退出都会同步并释放缓冲区、stream和TensorRT对象。

本地单元测试只能证明调用和资源生命周期逻辑。Gate B最终通过必须由Orin NX使用V7板端engine连续完成`60 -> 900 -> 60`实测，中间不reboot、不安装软件。

V7参考基线为900/900帧、8.32 FPS、`total_ms`均值118.00 ms、P95 188.14 ms、有效目标率100%、稳态RSS约1057 MB。场景不要求严格复现，性能只用于发现明显回退，不作为跨场景精确对齐指标。

## 2. 可直接交给板端agent的提示词

```text
你在Jetson Orin NX的 ~/Higgins/Visual-Deploy 仓库执行Visual-Deploy Gate B验收。
先完整阅读 docs/gate_b_cudart_board_agent.md，再严格执行其中第3至第8节。

约束：
1. 禁止apt、pip、conda、JetPack、CUDA、TensorRT、PyTorch、驱动和固件的安装、卸载或升级。
2. 禁止git reset、git checkout、git clean、递归删除、覆盖现有engine，禁止为了消除warning而屏蔽stderr。
3. 先确认Git工作树干净，再git fetch origin与git pull --ff-only；若不干净、不能快进、变量文件/本地配置/板端engine缺失，立即停止并报告。
4. 必须复用runs/device_engines/orin_nx中的V7板端engine，并核对文档给出的完整SHA-256；不得重新构建engine。
5. 按v8_gate_b_smoke_before、v8_gate_b_900、v8_gate_b_smoke_after连续运行60、900、60帧，中间不得reboot。
6. 任一段非complete、帧数或退出码错误，或出现PyTorch sm_87、跨设备engine、NvMap、OOM、CUDA初始化、段错误、Traceback、TensorRT error时，停止后续步骤并原样回报日志，不得自行清理或杀进程。
7. 通过后把三段报告复制到test/thread_profiles_engine/V8_GATE_B，补充GATE_B_RESULT.md，只暂存该目录并提交。不得提交engine、ONNX、本地配置、源码或runs目录。
8. 最终回复必须给出：同步后的Git SHA、三段manifest结果、900帧核心指标、V7/V8对比、warning扫描、torch_loaded检查、engine哈希、运行后残留进程检查、环境零修改声明、报告提交SHA。
```

## 3. 安全同步和前置检查

以下步骤沿用Gate A生成的变量文件。任何检查失败都停止，不猜路径、不删除本地文件：

```bash
cd "$HOME/Higgins/Visual-Deploy"
test -f runs/jetson_agent_no_env_fix.vars
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

git status --short --branch
test -z "$(git status --porcelain)"
git fetch origin
git pull --ff-only
git status --short --branch

test -x "$PYTHON_BIN"
test -f "$LOCAL_CONFIG"
test -f "$ENGINE_DIR/yolov10_sam_robust_standard.engine"
test -f "$ENGINE_DIR/rgbd_gcnet_l03_robustft.engine"
test ! -e runs/thread_profiles/v8_gate_b_smoke_before
test ! -e runs/thread_profiles/v8_gate_b_900
test ! -e runs/thread_profiles/v8_gate_b_smoke_after
```

`git pull --ff-only`不会删除被`.gitignore`忽略的本地engine、配置或变量文件。如果工作树不干净或无法快进，不继续同步，先把`git status --short --branch`原样回报。

## 4. 证明运行时不再依赖PyTorch

```bash
cd "$PROJECT_DIR"

if git grep -nE '^[[:space:]]*(from[[:space:]]+torch|import[[:space:]]+torch)|torch\.' -- visual_deploy/inference; then
  echo "FAIL: deployment inference still depends on torch" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY'
import sys
from visual_deploy.inference.cuda_runtime import CudaRuntime
from visual_deploy.inference.session import TensorRTEngineSession

runtime = CudaRuntime(0)
stream = runtime.create_stream()
stream.synchronize()
stream.close()
print("cuda_device", runtime.device_id)
print("torch_loaded", "torch" in sys.modules)
assert "torch" not in sys.modules
PY

ldconfig -p | grep 'libcudart.so'
sha256sum \
  "$ENGINE_DIR/yolov10_sam_robust_standard.engine" \
  "$ENGINE_DIR/rgbd_gcnet_l03_robustft.engine"
```

两份engine哈希必须分别为：

- YOLO：`b489dafaf3620cc4d4628df8111403d53d3863a5df29c5f0d270c5aa4485e894`
- GCNet：`64afd9a619aacb4d7b67a848340512ce6d77442903fc634d0da8921787f4121d`

## 5. 连续运行60、900、60帧

连接D435i，固定相机和当前场景。每条命令成功结束后再执行下一条，中间不reboot：

```bash
cd "$PROJECT_DIR"

"$PYTHON_BIN" scripts/collect_thread_profile.py \
  --mode realtime --config "$LOCAL_CONFIG" \
  --frames 60 --warmup-frames 10 \
  --session-name v8_gate_b_smoke_before

"$PYTHON_BIN" scripts/collect_thread_profile.py \
  --mode realtime --config "$LOCAL_CONFIG" \
  --frames 900 --warmup-frames 60 \
  --session-name v8_gate_b_900

"$PYTHON_BIN" scripts/collect_thread_profile.py \
  --mode realtime --config "$LOCAL_CONFIG" \
  --frames 60 --warmup-frames 10 \
  --session-name v8_gate_b_smoke_after
```

后置60帧是资源释放验收。它必须在900帧进程正常退出后立即成功，不能靠reboot恢复。

## 6. 自动检查和V7对比

```bash
cd "$PROJECT_DIR"
export GATE_B_RUN_ROOT="$PROJECT_DIR/runs/thread_profiles"
export GATE_B_V7_SUMMARY="$PROJECT_DIR/test/thread_profiles_engine/V7_DEVICE_BUILT/summary.json"

"$PYTHON_BIN" - <<'PY'
import json
import os
import re
from pathlib import Path

root = Path(os.environ["GATE_B_RUN_ROOT"])
runs = {
    "v8_gate_b_smoke_before": 60,
    "v8_gate_b_900": 900,
    "v8_gate_b_smoke_after": 60,
}
bad = re.compile(
    r"Found GPU0 Orin|compute capability.*8\.7|No published PyTorch CUDA builds|"
    r"engine plan file across different models|NvMap|out of memory|CUDA initialization|"
    r"Segmentation fault|Traceback|\[TRT\] \[E\]",
    re.IGNORECASE,
)

failures = []
for name, expected in runs.items():
    run = root / name
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    checks = {
        "status": manifest.get("status") == "complete",
        "frame_count": manifest.get("frame_count") == expected,
        "expected_frame_count": manifest.get("expected_frame_count") == expected,
        "child_return_code": manifest.get("child_return_code") == 0,
        "exit_code": manifest.get("exit_code") == 0,
    }
    for log_name in ("child_stdout.log", "child_stderr.log"):
        text = (run / log_name).read_text(encoding="utf-8", errors="replace")
        matches = sorted(set(match.group(0) for match in bad.finditer(text)))
        checks[f"{log_name}_clean"] = not matches
        if matches:
            print(name, log_name, "matches", matches)
    print(name, checks)
    failures.extend(f"{name}:{key}" for key, passed in checks.items() if not passed)

v7 = json.loads(Path(os.environ["GATE_B_V7_SUMMARY"]).read_text(encoding="utf-8"))
v8 = json.loads((root / "v8_gate_b_900" / "summary.json").read_text(encoding="utf-8"))
metrics = {
    "effective_fps": (v7["effective_fps"], v8["effective_fps"]),
    "valid_target_rate": (v7["valid_target_rate"], v8["valid_target_rate"]),
    "total_mean_ms": (v7["timings"]["total_ms"]["mean"], v8["timings"]["total_ms"]["mean"]),
    "total_p95_ms": (v7["timings"]["total_ms"]["p95"], v8["timings"]["total_ms"]["p95"]),
    "rss_p95_mb": (
        v7["process_resources"]["process_rss_mb"]["p95"],
        v8["process_resources"]["process_rss_mb"]["p95"],
    ),
}
for name, (before, after) in metrics.items():
    delta = (after / before - 1.0) * 100.0 if before else 0.0
    print(f"{name}: V7={before:.3f}, V8={after:.3f}, delta={delta:+.2f}%")

if v8["effective_fps"] < v7["effective_fps"] * 0.85:
    failures.append("v8_gate_b_900:effective_fps_regressed_over_15_percent")
if v8["timings"]["total_ms"]["mean"] > v7["timings"]["total_ms"]["mean"] * 1.15:
    failures.append("v8_gate_b_900:total_mean_regressed_over_15_percent")
if failures:
    raise SystemExit("GATE B FAIL: " + ", ".join(failures))
print("GATE B AUTOMATED CHECKS PASS")
PY

pgrep -af 'run_debug_viewer.py|run_realtime.py|collect_thread_profile.py' || true
```

15%阈值只拦截明显回退。果实数量、深度有效像素和功耗状态会改变耗时，所以报告仍要同时记录场景目标数量和V8实际指标，不要求机械复现V7每个数值。

## 7. Gate B通过条件

- 三段`manifest.json`均为`complete`，帧数正确，子进程与采集器退出码均为0；
- 三段日志均无PyTorch `sm_87`、跨设备engine、NvMap、OOM、CUDA初始化、段错误、Traceback或TensorRT error；
- 导入部署推理模块和初始化CUDA Runtime后，`torch_loaded False`；
- 900帧有效目标与检测/分割输出契约正常，FPS和平均总耗时相对V7无超过15%的明显回退；
- 900帧后可立即完成后置60帧，且没有残留viewer/realtime/profile进程；
- 板端engine哈希与V7一致，Jetson环境零修改。

如果只有warning消失，但后置60帧失败、需要reboot或出现资源错误，Gate B仍失败。

## 8. 保存报告并提交

先确认目标不存在，禁止覆盖历史报告：

```bash
cd "$PROJECT_DIR"
REPORT_TARGET="$PROJECT_DIR/test/thread_profiles_engine/V8_GATE_B"
test ! -e "$REPORT_TARGET"
mkdir -p "$REPORT_TARGET"
cp -a runs/thread_profiles/v8_gate_b_smoke_before "$REPORT_TARGET/smoke_before"
cp -a runs/thread_profiles/v8_gate_b_900 "$REPORT_TARGET/main_900"
cp -a runs/thread_profiles/v8_gate_b_smoke_after "$REPORT_TARGET/smoke_after"
```

在`$REPORT_TARGET/GATE_B_RESULT.md`写明Git SHA、三段manifest、V7/V8核心指标、warning扫描、`torch_loaded`、engine哈希、残留进程、环境零修改和最终判定。然后只提交报告目录：

```bash
git status --short
git add test/thread_profiles_engine/V8_GATE_B
git diff --cached --stat
git diff --cached --check
git diff --cached --name-only
```

暂存列表必须全部位于`test/thread_profiles_engine/V8_GATE_B/`。确认后：

```bash
git commit -m "test: add Orin torch-free Gate B report"
git status --short --branch
```

不要提交`runs/`、engine、ONNX、本地配置、变量文件或任何用户本地文件。
