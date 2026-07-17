# Findings & Decisions

## Requirements
- User has connected a RealSense camera and wants to test whether the code can run realtime.
- Verify by reading repo files first, then running bounded commands against the current workspace.

## Research Findings
- No local `AGENTS.md` file was found by `rg --files -g AGENTS.md`.
- Memory search found no relevant Visual-Deploy or RealSense notes.
- Relevant files found: `scripts/run_realtime.py`, `visual_deploy/camera/realsense_source.py`, `configs/deploy.yaml`, `README.md`, `docs/deployment.md`, and `tests/test_realsense_source.py`.
- Git status before work showed user-existing modifications in `pyproject.toml` and `requirements.txt`.
- `scripts/run_realtime.py` builds YOLO, GCNet, RealSense source, then loops over frames and supports `--max-frames`.
- `README.md` and `docs/deployment.md` document the realtime command as `.\.venv\Scripts\python.exe scripts\run_realtime.py --config configs\deploy.yaml --max-frames 30`.
- `configs/deploy.yaml` requests 640x480 at 30 FPS with aligned depth and CUDA device `cuda:0` for detection and segmentation.
- `GCNetSegmentor` falls back to CPU when CUDA is requested but unavailable; `UltralyticsYoloDetector` passes its configured `device` directly to model prediction.
- Environment check using `.venv` succeeded for `numpy 1.26.4`, `opencv-python 4.11.0`, `PyYAML 6.0.3`, `torch 2.0.1+cpu`, `ultralytics 8.1.34`, and `pyrealsense2 2.58.2`.
- Torch reports `torch.cuda.is_available False` and `torch.version.cuda None`; the environment is CPU-only.
- Both expected weights exist: `weights/yolo_detect.pt` and `weights/rgbd_gcnet_l03_robustft_inference.pt`.
- Camera-only RealSense source captured 10 aligned 640x480 frames. Initial startup-inclusive elapsed rate was 8.90 FPS, while RealSense timestamps after startup advanced at approximately 30 FPS.
- Default realtime command failed after camera startup because YOLO was configured with `device: cuda:0` but Torch reports zero CUDA devices.
- Full pipeline with in-memory CPU override processed 30 RealSense frames. Result: 6.674 seconds elapsed, 4.495 end-to-end FPS, first frame processing 2.342 seconds, after-first average processing 0.123 seconds per frame (8.118 FPS), and 0 valid targets (`no_valid_grasp_candidate` for all 30 frames).

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| Run bounded hardware tests | Prevents a realtime loop from running indefinitely in the shared session. |
| Inspect dependency and config files before running | Avoid guessing commands, package names, or flags. |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| PowerShell heredoc syntax failed before Python ran | Switched to PowerShell here-string piped into `.venv\Scripts\python.exe -`. |
| Default realtime command fails in current `.venv` | Root cause is config requests CUDA while installed Torch is CPU-only; test an in-memory CPU override to isolate pipeline behavior. |

## Resources
- `scripts/run_realtime.py`
- `visual_deploy/camera/realsense_source.py`
- `configs/deploy.yaml`
- `README.md`
- `docs/deployment.md`
- `tests/test_realsense_source.py`

## Visual/Browser Findings
- Not applicable.
