# Progress Log

## Session: 2026-07-03

### Phase 1: Discover Runtime Path
- **Status:** complete
- **Started:** 2026-07-03
- Actions taken:
  - Checked for planning catchup context.
  - Searched for local `AGENTS.md`; none found.
  - Searched memory registry for project and RealSense terms; no relevant notes found.
  - Listed repo files and searched for RealSense/realtime references.
  - Created planning files to track this hardware runtime check.
  - Read realtime script, RealSense source, config, dependency declarations, documentation excerpts, and model loader code.
  - Verified imports, Torch/CUDA state, and expected weights.
  - Captured 10 frames from `RealSenseSource` directly.
- Files created/modified:
  - `task_plan.md` (created)
  - `findings.md` (created)
  - `progress.md` (created)

## Test Results
| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Environment imports | `.venv` import check | Required modules import | All required modules imported | pass |
| Weights present | Check two configured weight files | Both exist | Both exist | pass |
| CUDA availability | Torch CUDA check | Know runtime device state | CPU-only Torch, CUDA unavailable | pass |
| Camera-only capture | `RealSenseSource` 10 frames | Aligned color/depth frames | 10 frames captured, 640x480 color/depth | pass |
| Default realtime command | `scripts\run_realtime.py --config configs\deploy.yaml --max-frames 3` | Process bounded frames or expose blocker | Failed at YOLO device selection: invalid CUDA device 0 | fail |
| CPU override realtime pipeline | In-memory `detection.device=cpu`, `segmentation.device=cpu`, 30 frames | Determine whether code path runs and estimate throughput | Processed 30 frames in 6.674s, 4.495 end-to-end FPS, 8.118 FPS after first-frame warmup | pass, below realtime |
| CPU debug viewer launch | `scripts\run_debug_viewer.py --config configs\deploy.cpu.local.yaml --labels` | Visual debug process stays running for user | Python process PID 23640 running, stdout shows live frame records and valid targets after startup | pass |

### Follow-up: CPU Debug Viewer
- **Status:** complete
- Actions taken:
  - Read `scripts/run_debug_viewer.py` and confirmed it reads device settings only from config.
  - Created `configs/deploy.cpu.local.yaml` with both `detection.device` and `segmentation.device` set to `cpu`.
  - Started the viewer as a detached Python process so the user can close it manually.
  - Verified process `23640` is still running and `runs\debug_viewer_cpu_logs\stdout.log` is updating.
- Files created/modified:
  - `configs/deploy.cpu.local.yaml` (created)
  - `runs\debug_viewer_cpu_logs\stdout.log` (created by running viewer)
  - `runs\debug_viewer_cpu_logs\stderr.log` (created by running viewer)
  - `progress.md` (updated)

## Error Log
| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-07-03 | PowerShell rejected `python - <<'PY'` heredoc syntax before Python started | 1 | Rerun using `@'...'@ | .\.venv\Scripts\python.exe -`. |
| 2026-07-03 | `ValueError: Invalid CUDA 'device=0' requested` from Ultralytics during realtime run | 1 | Identified CPU-only Torch with config requesting `cuda:0`; rerun with in-memory CPU device override. |

## 5-Question Reboot Check
| Question | Answer |
|----------|--------|
| Where am I? | Phase 4: Report |
| Where am I going? | Verify environment, run bounded realtime test, report result |
| What's the goal? | Verify whether connected RealSense runtime code can run realtime |
| What have I learned? | See `findings.md` |
| What have I done? | See this log |
