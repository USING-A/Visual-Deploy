from __future__ import annotations

import argparse
import copy
import json
import platform
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from visual_deploy.config import load_config, resolve_path
from visual_deploy.observability.profile_analysis import (
    build_profile_summary,
    load_jsonl,
    parse_tegrastats_line,
    write_jsonl,
    write_profile_reports,
)


def main() -> None:
    args = _parse_args()
    result = collect_profile(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["child_return_code"] != 0:
        raise SystemExit(int(result["child_return_code"]))


def collect_profile(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("psutil is required; install deployment requirements before profiling") from exc

    config_path = Path(args.config).resolve()
    source_config = load_config(config_path)
    output_root = Path(args.output_root).resolve()
    session_name = args.session_name or _default_session_name()
    session_dir = output_root / _validate_session_name(session_name)
    session_dir.mkdir(parents=True, exist_ok=False)

    generated_config = _build_profile_config(source_config, config_path, session_dir)
    profile_config_path = session_dir / "profile_config.yaml"
    with profile_config_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(generated_config, file, allow_unicode=True, sort_keys=False)

    command = _build_child_command(args, profile_config_path)
    tegrastats_command = _resolve_tegrastats(args.tegrastats, args.sample_interval_ms)
    manifest: dict[str, Any] = {
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "source_config": str(config_path),
        "profile_config": str(profile_config_path),
        "command": command,
        "tegrastats_command": tegrastats_command,
        "sample_interval_ms": args.sample_interval_ms,
        "warmup_frames": args.warmup_frames,
    }
    _write_json(session_dir / "manifest.json", manifest)

    child_stdout_path = session_dir / "child_stdout.log"
    child_stderr_path = session_dir / "child_stderr.log"
    tegrastats_raw_path = session_dir / "tegrastats.log"
    tegrastats_lines: list[dict[str, Any]] = []
    process_samples: list[dict[str, Any]] = []
    started = time.monotonic()

    tegra_process: subprocess.Popen[str] | None = None
    tegra_thread: threading.Thread | None = None
    with child_stdout_path.open("w", encoding="utf-8") as child_stdout, child_stderr_path.open(
        "w", encoding="utf-8"
    ) as child_stderr, tegrastats_raw_path.open("w", encoding="utf-8") as tegra_log:
        child = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=child_stdout,
            stderr=child_stderr,
            text=True,
        )
        process = psutil.Process(child.pid)
        process.cpu_percent(interval=None)

        if tegrastats_command:
            tegra_process = subprocess.Popen(
                tegrastats_command,
                cwd=PROJECT_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert tegra_process.stdout is not None
            tegra_thread = threading.Thread(
                target=_read_tegrastats,
                args=(tegra_process.stdout, tegra_log, tegrastats_lines, started),
                daemon=True,
            )
            tegra_thread.start()

        timed_out = False
        try:
            while child.poll() is None:
                elapsed_s = time.monotonic() - started
                if args.timeout_s is not None and elapsed_s >= args.timeout_s:
                    timed_out = True
                    _terminate_process(child)
                    break
                process_samples.append(_sample_process(process, psutil, elapsed_s))
                try:
                    child.wait(timeout=args.sample_interval_ms / 1000.0)
                except subprocess.TimeoutExpired:
                    pass
        except KeyboardInterrupt:
            manifest["interrupted"] = True
            _terminate_process(child)
        finally:
            child_return_code = child.wait()
            if tegra_process is not None:
                _terminate_process(tegra_process)
            if tegra_thread is not None:
                tegra_thread.join(timeout=5.0)

    duration_s = time.monotonic() - started
    tegrastats_samples = []
    for record in tegrastats_lines:
        parsed = parse_tegrastats_line(str(record["raw"]))
        parsed["elapsed_s"] = float(record["elapsed_s"])
        tegrastats_samples.append(parsed)

    write_jsonl(session_dir / "process_resources.jsonl", process_samples)
    write_jsonl(session_dir / "tegrastats.jsonl", tegrastats_samples)
    timing_path = _find_timing_file(session_dir / "pipeline_runs")
    timing_records = load_jsonl(timing_path) if timing_path is not None else []
    summary = build_profile_summary(
        timing_records,
        process_samples,
        tegrastats_samples,
        warmup_frames=args.warmup_frames,
        target_fps=float(generated_config.get("camera", {}).get("fps", 30.0)),
    )
    summary["collection_duration_s"] = duration_s
    summary["collection_fps_including_startup"] = len(timing_records) / duration_s if duration_s > 0.0 else 0.0
    write_profile_reports(session_dir, summary)

    manifest.update(
        {
            "status": "timed_out"
            if timed_out
            else ("interrupted" if manifest.get("interrupted") else ("complete" if child_return_code == 0 else "child_failed")),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "duration_s": duration_s,
            "child_return_code": child_return_code,
            "timings_path": str(timing_path) if timing_path else None,
            "frame_count": len(timing_records),
            "process_sample_count": len(process_samples),
            "tegrastats_sample_count": len(tegrastats_samples),
        }
    )
    _write_json(session_dir / "manifest.json", manifest)
    return {
        "session_dir": str(session_dir),
        "status": manifest["status"],
        "child_return_code": child_return_code,
        "frame_count": len(timing_records),
        "report": str(session_dir / "report.md"),
    }


def _build_profile_config(config: dict[str, Any], config_path: Path, session_dir: Path) -> dict[str, Any]:
    generated = copy.deepcopy(config)
    for section in ("detection", "segmentation"):
        weights = generated.get(section, {}).get("weights")
        if weights:
            generated[section]["weights"] = str(resolve_path(config_path, weights).resolve())
    generated["profiling"] = {"enabled": True}
    generated["debug"] = {"enabled": False}
    generated["diagnostics"] = {"enabled": False}
    recording = generated.setdefault("recording", {})
    recording.update(
        {
            "enabled": True,
            "output_root": str((session_dir / "pipeline_runs").resolve()),
            "save_detections": False,
            "save_candidates": False,
            "save_targets": False,
            "save_errors": False,
            "save_timings": True,
            "save_events": False,
            "save_frames": False,
            "save_depth": False,
            "save_masks": False,
            "save_overlays": False,
            "save_event_artifacts": False,
        }
    )
    return generated


def _build_child_command(args: argparse.Namespace, config_path: Path) -> list[str]:
    if args.mode == "realtime":
        return [
            str(args.python),
            str(PROJECT_ROOT / "scripts" / "run_realtime.py"),
            "--config",
            str(config_path),
            "--max-frames",
            str(args.frames),
        ]
    missing = [name for name in ("rgb", "depth", "fx", "fy", "ppx", "ppy") if getattr(args, name) is None]
    if missing:
        raise ValueError(f"offline mode requires: {', '.join('--' + name for name in missing)}")
    command = [
        str(args.python),
        str(PROJECT_ROOT / "scripts" / "run_offline_smoke.py"),
        "--config",
        str(config_path),
        "--rgb",
        str(Path(args.rgb).resolve()),
        "--depth",
        str(Path(args.depth).resolve()),
        "--fx",
        str(args.fx),
        "--fy",
        str(args.fy),
        "--ppx",
        str(args.ppx),
        "--ppy",
        str(args.ppy),
        "--repeat-frames",
        str(args.frames),
    ]
    if args.use_mock_models:
        command.append("--use-mock-models")
    return command


def _resolve_tegrastats(value: str, interval_ms: int) -> list[str] | None:
    if value.lower() == "off":
        return None
    executable = shutil.which("tegrastats") if value.lower() == "auto" else str(Path(value).expanduser())
    if not executable:
        return None
    if value.lower() != "auto" and not Path(executable).is_file():
        raise FileNotFoundError(f"tegrastats executable not found: {executable}")
    return [executable, "--interval", str(interval_ms)]


def _sample_process(process: Any, psutil: Any, elapsed_s: float) -> dict[str, float]:
    try:
        memory = process.memory_info()
        return {
            "elapsed_s": elapsed_s,
            "process_cpu_percent": float(process.cpu_percent(interval=None)),
            "process_rss_mb": float(memory.rss) / (1024.0 * 1024.0),
            "system_cpu_percent": float(psutil.cpu_percent(interval=None)),
            "system_ram_percent": float(psutil.virtual_memory().percent),
        }
    except psutil.Error:
        return {"elapsed_s": elapsed_s}


def _read_tegrastats(stream: TextIO, log: TextIO, records: list[dict[str, Any]], started: float) -> None:
    for line in stream:
        stripped = line.strip()
        if not stripped:
            continue
        log.write(stripped + "\n")
        log.flush()
        records.append({"elapsed_s": time.monotonic() - started, "raw": stripped})


def _find_timing_file(root: Path) -> Path | None:
    paths = sorted(root.glob("*/timings.jsonl")) if root.exists() else []
    if len(paths) > 1:
        raise RuntimeError(f"expected one timings.jsonl, found {len(paths)} under {root}")
    return paths[0] if paths else None


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5.0)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _default_session_name() -> str:
    return datetime.now(timezone.utc).strftime("thread_profile_%Y%m%d_%H%M%S_%f")


def _validate_session_name(value: str) -> str:
    if value in {"", ".", ".."} or any(character in value for character in "/\\:"):
        raise ValueError("session_name must be a single relative directory name")
    return value


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect deployment timing/resources and generate threading optimization reports."
    )
    parser.add_argument("--mode", choices=("realtime", "offline"), default="realtime")
    parser.add_argument("--config", default="configs/deploy.yaml")
    parser.add_argument("--frames", type=_positive_int, default=300)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--sample-interval-ms", type=_positive_int, default=1000)
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--output-root", default="runs/thread_profiles")
    parser.add_argument("--session-name", default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--tegrastats", default="auto", help="auto, off, or an explicit tegrastats executable")
    parser.add_argument("--rgb", type=Path)
    parser.add_argument("--depth", type=Path)
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--ppx", type=float)
    parser.add_argument("--ppy", type=float)
    parser.add_argument("--use-mock-models", action="store_true")
    args = parser.parse_args()
    if args.warmup_frames < 0:
        parser.error("--warmup-frames must be non-negative")
    if args.timeout_s is not None and args.timeout_s <= 0.0:
        parser.error("--timeout-s must be positive")
    return args


if __name__ == "__main__":
    main()
