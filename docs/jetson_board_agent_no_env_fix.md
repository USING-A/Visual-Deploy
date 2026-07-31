# Jetson板端agent操作指南：不修改系统环境修复部署警告

本文用于把操作任务交给运行在Jetson Orin NX上的另一个agent。目标是在不安装、卸载或升级任何Jetson软件包的前提下：

1. 在目标Orin NX上从ONNX重新生成TensorRT engine，消除跨设备engine警告；
2. 保留仓库中现有的模型、engine、配置和本地文件；
3. 收集一份可追溯的900帧验收报告；
4. 使用当前torch-free TensorRT运行时完成独立的Gate B板端验收。

## 1. 可直接交给板端agent的任务说明

将下面这段话连同本文路径交给板端agent：

> Gate A已经完成，本文第3节和第7节保留为历史操作与证据。执行Gate B时不要重复构建engine，改为严格按照`docs/gate_b_cudart_board_agent.md`中的提示词和`60 -> 900 -> 60`流程执行。不得修改Jetson环境，不得删除、覆盖或提交本地engine和配置。

## 2. 操作边界

允许：

- 读取系统和Git状态；
- 在 `runs/` 下创建板端专用engine、配置、日志和报告；
- 使用仓库现有脚本和系统已有的 `trtexec`；
- 运行短测试和900帧实时采集；
- 将报告复制到 `test/thread_profiles_engine/` 后提交。

禁止：

- `apt install/remove/upgrade`；
- `pip install/uninstall`；
- `conda install/remove/update`；
- 修改JetPack、L4T、CUDA、TensorRT、PyTorch、驱动或固件；
- 设置warning过滤器掩盖报错；
- 设置 `TORCH_CUDA_ARCH_LIST=8.7` 并声称修复了预编译wheel；
- 覆盖或提交仓库跟踪的 `weights/*.engine`；
- 删除、重置或丢弃任何本地文件；
- 未经用户确认自动结束残留进程或重启设备。

## 3. Gate A：在当前Orin NX上重新生成engine

### 3.1 固定工作目录和当前Python

V6使用的是下面的Python。先验证路径，不要创建新环境：

```bash
set -o pipefail

PROJECT_DIR="$HOME/Higgins/Visual-Deploy"
PYTHON_BIN="/home/liancheng/miniconda3/envs/visual/bin/python"
TRTEXEC_BIN="/usr/src/tensorrt/bin/trtexec"
ENGINE_DIR="$PROJECT_DIR/runs/device_engines/orin_nx"
LOCAL_CONFIG="$PROJECT_DIR/runs/deploy_orin_device_engine.yaml"
STATE_FILE="$PROJECT_DIR/runs/jetson_agent_no_env_fix.vars"

cd "$PROJECT_DIR"
test -x "$PYTHON_BIN"
test -x "$TRTEXEC_BIN"
mkdir -p "$PROJECT_DIR/runs"
printf '%s\n' \
  'set -o pipefail' \
  'PROJECT_DIR="$HOME/Higgins/Visual-Deploy"' \
  'PYTHON_BIN="/home/liancheng/miniconda3/envs/visual/bin/python"' \
  'TRTEXEC_BIN="/usr/src/tensorrt/bin/trtexec"' \
  'ENGINE_DIR="$PROJECT_DIR/runs/device_engines/orin_nx"' \
  'LOCAL_CONFIG="$PROJECT_DIR/runs/deploy_orin_device_engine.yaml"' \
  > "$STATE_FILE"
pwd
```

如果任一 `test -x` 失败，停止并报告实际的 `command -v python` 和 `command -v trtexec`，不要安装任何东西。

板端agent的每次shell调用都可能是新进程。后续每个命令块执行前都先运行：

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"
```

### 3.2 只读检查，不处理脏文件和残留进程

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

git status --short --branch
git rev-parse HEAD
git log -1 --oneline

pgrep -af 'run_debug_viewer.py|run_realtime.py|collect_thread_profile.py' || true

uname -a
cat /etc/nv_tegra_release
dpkg-query -W nvidia-l4t-core nvidia-jetpack libnvinfer-bin 2>&1 || true
"$TRTEXEC_BIN" --version
free -h
df -h .
```

停止条件：

- `git status` 显示任何已有修改或未跟踪的重要文件；
- `pgrep` 找到正在运行的viewer、实时推理或采集进程；
- 剩余磁盘空间不足以同时容纳约100 MB临时文件和日志；
- 当前目录不是预期仓库。

发现以上情况时只报告，不执行 `reset`、`checkout`、`clean`、删除或杀进程。

工作树干净且用户要求同步到远端最新提交时，只允许快进同步：

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

git fetch origin
git pull --ff-only
git status --short --branch
git rev-parse HEAD
```

`git pull --ff-only` 失败时停止并报告分支状态，不要合并、变基或重置。

### 3.3 记录构建前来源和哈希

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

test -s weights/yolov10_sam_robust_standard.onnx
test -s weights/rgbd_gcnet_l03_robustft.onnx
test -s weights/yolov10_sam_robust_standard.engine
test -s weights/rgbd_gcnet_l03_robustft.engine

sha256sum \
  weights/yolov10_sam_robust_standard.onnx \
  weights/rgbd_gcnet_l03_robustft.onnx \
  weights/yolov10_sam_robust_standard.engine \
  weights/rgbd_gcnet_l03_robustft.engine
```

不要覆盖上述两个受Git跟踪的engine。Orin专用engine写入忽略目录：

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

test ! -e "$ENGINE_DIR" || {
  echo "STOP: $ENGINE_DIR already exists; preserve it and choose a new directory only after user approval"
  exit 2
}
mkdir -p "$ENGINE_DIR"
```

如果该目录已经存在，停止并报告。不要删除或覆盖旧文件。由用户决定是否换一个新目录名。

### 3.4 构建两个板端专用FP16 engine

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

"$PYTHON_BIN" scripts/build_tensorrt_engines.py \
  --trtexec "$TRTEXEC_BIN" \
  --yolo-onnx weights/yolov10_sam_robust_standard.onnx \
  --gcnet-onnx weights/rgbd_gcnet_l03_robustft.onnx \
  --yolo-engine "$ENGINE_DIR/yolov10_sam_robust_standard.engine" \
  --gcnet-engine "$ENGINE_DIR/rgbd_gcnet_l03_robustft.engine" \
  2>&1 | tee "$ENGINE_DIR/build.log"
```

构建脚本会先写临时同级文件，仅在 `trtexec` 成功且产物非空后替换目标。命令返回非零、日志包含TensorRT错误或任一输出为空时，停止并保留全部日志。

成功后记录：

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

test -s "$ENGINE_DIR/yolov10_sam_robust_standard.engine"
test -s "$ENGINE_DIR/rgbd_gcnet_l03_robustft.engine"
ls -lh "$ENGINE_DIR"
sha256sum "$ENGINE_DIR"/*.engine | tee "$ENGINE_DIR/engine_sha256.txt"
```

### 3.5 生成不会污染Git的Jetson本地配置

下面的配置写入已被 `.gitignore` 忽略的 `runs/`，原始 `configs/deploy.yaml` 不会被修改：

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

"$PYTHON_BIN" - "$PROJECT_DIR/configs/deploy.yaml" "$LOCAL_CONFIG" "$ENGINE_DIR" <<'PY'
from pathlib import Path
import sys
import yaml

source = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
engine_dir = Path(sys.argv[3]).resolve()

with source.open("r", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

config["detection"]["weights"] = str(engine_dir / "yolov10_sam_robust_standard.engine")
config["segmentation"]["weights"] = str(engine_dir / "rgbd_gcnet_l03_robustft.engine")

output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=False)

print(output)
PY

grep -nE 'weights:|backend:|reuse_buffers:' "$LOCAL_CONFIG"
git status --short --branch
```

此时Git工作树仍应保持干净。若出现 `weights/*.engine` 或 `configs/deploy.yaml` 被修改，停止并报告，不要还原或提交这些文件。

### 3.6 先跑60帧短测试

连接D435i并保持场景静止：

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

"$PYTHON_BIN" scripts/collect_thread_profile.py \
  --mode realtime \
  --config "$LOCAL_CONFIG" \
  --frames 60 \
  --warmup-frames 10 \
  --session-name v7_device_engine_smoke
```

检查：

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

cat runs/thread_profiles/v7_device_engine_smoke/manifest.json

grep -nE \
  'engine plan file across different models|NvMap|out of memory|CUDA initialization|Segmentation fault|Traceback|\[TRT\] \[E\]' \
  runs/thread_profiles/v7_device_engine_smoke/child_stdout.log \
  runs/thread_profiles/v7_device_engine_smoke/child_stderr.log || true
```

Gate A短测试通过条件：

- `manifest.json` 中 `status` 为 `complete`；
- `frame_count` 和 `expected_frame_count` 均为60；
- `child_return_code` 和 `exit_code` 均为0；
- 不再出现 `engine plan file across different models`；
- 无NvMap、OOM、CUDA初始化、段错误、Traceback和TensorRT error。

当前代码仍导入PyTorch时，`sm_87` warning预计仍会出现。它不属于Gate A失败，但必须保留在报告中。

### 3.7 运行900帧V7验收

短测试通过后执行：

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

"$PYTHON_BIN" scripts/collect_thread_profile.py \
  --mode realtime \
  --config "$LOCAL_CONFIG" \
  --frames 900 \
  --warmup-frames 60 \
  --session-name v7_device_engine_900
```

记录来源信息：

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

{
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_status_begin"
  git status --short --branch
  echo "git_status_end"
  echo "platform=$(uname -a)"
  echo "l4t_begin"
  cat /etc/nv_tegra_release
  echo "l4t_end"
  "$TRTEXEC_BIN" --version
  "$PYTHON_BIN" - <<'PY'
import sys
import tensorrt
import torch
print("python", sys.version.replace("\n", " "))
print("tensorrt", tensorrt.__version__)
print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
print("device_capability", torch.cuda.get_device_capability())
print("compiled_arches", torch.cuda.get_arch_list())
PY
  sha256sum \
    weights/yolov10_sam_robust_standard.onnx \
    weights/rgbd_gcnet_l03_robustft.onnx \
    "$ENGINE_DIR/yolov10_sam_robust_standard.engine" \
    "$ENGINE_DIR/rgbd_gcnet_l03_robustft.engine"
} > "$PROJECT_DIR/runs/thread_profiles/v7_device_engine_900/provenance.txt" 2>&1

cat "$PROJECT_DIR/runs/thread_profiles/v7_device_engine_900/manifest.json"
cat "$PROJECT_DIR/runs/thread_profiles/v7_device_engine_900/report.md"
```

V7通过条件：

- 900/900帧完成，退出码0；
- 不出现跨设备engine、NvMap、OOM、CUDA初始化、段错误或TensorRT error；
- 稳态RSS形成平台，不随运行时间持续增长；
- 有效目标率、平均/P95总延迟、frame age、GPU利用率和温度均写入报告；
- 关闭后能够立即再次完成一次60帧短测试，无需reboot。

## 4. Gate B：验证torch-free TensorRT运行时

Gate B代码侧已改为直接使用JetPack自带的CUDA Runtime管理缓冲区、异步拷贝和stream，部署运行时不再导入PyTorch。板端完整命令、自动判定规则、报告回传方式和可复制给另一个agent的提示词统一放在：

- [`docs/gate_b_cudart_board_agent.md`](gate_b_cudart_board_agent.md)

必须使用V7已经在本机生成的两份engine，连续执行`60 -> 900 -> 60`，中间不得reboot。只有三段均完成、错误扫描为空、后置60帧可立即启动，才能确认CUDA资源释放与Gate B设备验收通过。

## 5. 回传报告

不要提交 `runs/device_engines/` 中的设备专用engine，也不要提交本地配置。仅复制采集报告：

```bash
. "$HOME/Higgins/Visual-Deploy/runs/jetson_agent_no_env_fix.vars"
cd "$PROJECT_DIR"

REPORT_TARGET="$PROJECT_DIR/test/thread_profiles_engine/V7_DEVICE_BUILT"
test ! -e "$REPORT_TARGET"
cp -a "$PROJECT_DIR/runs/thread_profiles/v7_device_engine_900" "$REPORT_TARGET"

git status --short
git add test/thread_profiles_engine/V7_DEVICE_BUILT
git diff --cached --stat
git diff --cached --check
```

确认暂存区只包含报告后，再创建提交：

```bash
git commit -m "test: add Orin device-built engine report"
```

如果暂存区包含engine、ONNX、配置、源码或其他用户文件，停止并报告，不要提交。

回传内容必须包括：

1. 新提交SHA；
2. `manifest.json`；
3. `report.md`；
4. `provenance.txt`；
5. warning/error扫描结果；
6. engine构建日志路径与两个engine哈希；
7. 是否完成关闭后立即重启测试；
8. 明确说明Jetson环境是否保持不变。

## 6. 判定规则

| 状态 | 含义 |
|---|---|
| Gate A通过、Gate B未执行 | 跨设备engine问题已修复；PyTorch警告等待代码侧torch-free运行时 |
| Gate A和Gate B均通过 | 两类警告均在不修改Jetson环境的条件下修复 |
| Gate A失败 | engine、TensorRT、进程、相机或资源仍有阻塞，不得继续900帧验收 |
| 仅隐藏warning | 不算修复 |

---

## 7. 执行记录与当前状态

**执行日期**: 2026-07-31

### Gate A ✅ 通过

| 检查项 | 结果 |
|---|---|
| 3.1 路径验证 | `python` 和 `trtexec` 均可执行 |
| 3.2 只读检查 | Git 干净，无残留进程，磁盘 171G 可用 |
| 3.3 ONNX/engine 哈希 | ONNX 匹配契约，现有 engine 未修改 |
| 3.4 板端 engine 构建 | YOLO 510s / 7.1 MiB，GCNet 532s / 21.2 MiB |
| 3.5 本地配置 | 写入 `runs/deploy_orin_device_engine.yaml`，Git 干净 |
| 3.6 60帧短测试 | `complete`，60/60 帧，exit 0，**无跨设备 engine 警告** |
| 3.7 900帧验收 | `complete`，900/900 帧，exit 0 |

### V7 性能

| 指标 | 值 |
|---|---|
| P50 total | 105.9 ms |
| FPS | 8.32 |
| 有效目标率 | 100% |
| confirmed_tracks | 2 |
| fast_path | 96.7% |
| 拒绝率 | 0% |

### 警告扫描

- ❌ 跨设备 engine 警告：**已消除**
- ❌ NvMap / OOM / Traceback / TRT error：**无**
- ⚠️ PyTorch `sm_87` 警告：**仍存在**（非 Gate A 失败条件）

### Engine 位置与哈希

| Engine | 路径 | SHA-256 |
|---|---|---|
| YOLO | `runs/device_engines/orin_nx/yolov10_sam_robust_standard.engine` | `b489dafaf362...` |
| GCNet | `runs/device_engines/orin_nx/rgbd_gcnet_l03_robustft.engine` | `64afd9a619aa...` |

构建日志：`runs/device_engines/orin_nx/build.log`

### Gate B历史状态：V7时阻塞，现等待V8板端验收

V7采集时，`visual_deploy/inference/session.py` 仍包含PyTorch CUDA调用，因此报告中的`sm_87`警告是当时真实状态。当前仓库已实现torch-free cudart运行时；不要改写V7历史报告，按独立V8 Gate B流程重新采集证据。

### 提交

- `6cb5b3d` — `test: add Orin device-built engine report (V7 Gate A)`
- 报告位置：`test/thread_profiles_engine/V7_DEVICE_BUILT/`

### Jetson 环境

**零次 apt / pip / conda 操作。环境完全未修改。**
