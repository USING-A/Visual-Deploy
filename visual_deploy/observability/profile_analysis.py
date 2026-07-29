from __future__ import annotations

import csv
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from statistics import fmean
from typing import Any


_RAM_RE = re.compile(r"\bRAM\s+(\d+)/(\d+)MB\b")
_CPU_RE = re.compile(r"\bCPU\s*\[([^]]+)]")
_GPU_RE = re.compile(r"\bGR3D_FREQ\s+(\d+(?:\.\d+)?)%")
_TEMP_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)@(-?\d+(?:\.\d+)?)C\b")
_POWER_RE = re.compile(r"\b([A-Za-z][A-Za-z0-9_]*)\s+(\d+(?:\.\d+)?)mW/(\d+(?:\.\d+)?)mW\b")
_CPU_LOAD_RE = re.compile(r"(\d+(?:\.\d+)?)%@")
_PREFERRED_INPUT_RAILS = ("VDD_IN", "VIN_SYS_5V0", "SYS5V", "POM_5V_IN")
_NON_METRIC_FIELDS = {"frame_id", "timestamp_ms", "completed_monotonic_ms", "elapsed_s"}


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def write_jsonl(path: str | Path, records: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    with target.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(dict(record), ensure_ascii=False) + "\n")


def parse_tegrastats_line(line: str) -> dict[str, float]:
    sample: dict[str, float] = {}
    ram = _RAM_RE.search(line)
    if ram:
        used_mb, total_mb = (float(value) for value in ram.groups())
        sample["system_ram_used_mb"] = used_mb
        sample["system_ram_total_mb"] = total_mb
        sample["system_ram_percent"] = used_mb / total_mb * 100.0 if total_mb else 0.0

    cpu = _CPU_RE.search(line)
    if cpu:
        loads = [float(value) for value in _CPU_LOAD_RE.findall(cpu.group(1))]
        if loads:
            sample["cpu_mean_percent"] = fmean(loads)
            sample["cpu_max_core_percent"] = max(loads)
            sample["cpu_active_cores"] = float(len(loads))

    gpu = _GPU_RE.search(line)
    if gpu:
        sample["gpu_util_percent"] = float(gpu.group(1))

    temperatures = {name.lower(): float(value) for name, value in _TEMP_RE.findall(line)}
    for name, value in temperatures.items():
        sample[f"temperature_{name}_c"] = value
    if temperatures:
        sample["temperature_max_c"] = max(temperatures.values())

    rails = {name: (float(current), float(average)) for name, current, average in _POWER_RE.findall(line)}
    for name, (current, average) in rails.items():
        key = name.lower()
        sample[f"power_{key}_mw"] = current
        sample[f"power_{key}_average_mw"] = average
    for preferred in _PREFERRED_INPUT_RAILS:
        if preferred in rails:
            sample["input_power_mw"] = rails[preferred][0]
            sample["input_power_average_mw"] = rails[preferred][1]
            break
    return sample


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    values_by_metric: dict[str, list[float]] = {}
    for record in records:
        for key, value in _flatten_numeric(record).items():
            values_by_metric.setdefault(key, []).append(value)
    return {key: summarize_values(values) for key, values in sorted(values_by_metric.items())}


def summarize_values(values: Sequence[float]) -> dict[str, float]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {}
    return {
        "count": float(len(finite)),
        "mean": fmean(finite),
        "min": finite[0],
        "p50": percentile(finite, 50.0),
        "p95": percentile(finite, 95.0),
        "p99": percentile(finite, 99.0),
        "max": finite[-1],
    }


def percentile(sorted_values: Sequence[float], percent: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= percent <= 100.0:
        raise ValueError("percent must be between 0 and 100")
    position = (len(sorted_values) - 1) * percent / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction)


def build_profile_summary(
    timing_records: Sequence[Mapping[str, Any]],
    process_records: Sequence[Mapping[str, Any]],
    tegrastats_records: Sequence[Mapping[str, Any]],
    *,
    warmup_frames: int = 0,
    target_fps: float = 30.0,
) -> dict[str, Any]:
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    selected_timings = list(timing_records[warmup_frames:])
    valid_values = [bool(record.get("valid_target", False)) for record in selected_timings]
    completion_times = [
        float(record["completed_monotonic_ms"])
        for record in selected_timings
        if isinstance(record.get("completed_monotonic_ms"), (int, float))
    ]
    effective_fps = 0.0
    if len(completion_times) >= 2 and completion_times[-1] > completion_times[0]:
        effective_fps = (len(completion_times) - 1) * 1000.0 / (completion_times[-1] - completion_times[0])
    summary: dict[str, Any] = {
        "frame_count_total": len(timing_records),
        "warmup_frames_excluded": min(warmup_frames, len(timing_records)),
        "frame_count_analyzed": len(selected_timings),
        "target_fps": float(target_fps),
        "target_frame_interval_ms": 1000.0 / target_fps if target_fps > 0.0 else 0.0,
        "valid_target_rate": (sum(valid_values) / len(valid_values)) if valid_values else 0.0,
        "effective_fps": effective_fps,
        "timings": summarize_records(selected_timings),
        "process_resources": summarize_records(process_records),
        "tegrastats": summarize_records(tegrastats_records),
    }
    summary["recommendations"] = make_threading_recommendations(summary)
    return summary


def make_threading_recommendations(summary: Mapping[str, Any]) -> list[str]:
    recommendations: list[str] = []
    timings = summary.get("timings", {})
    if int(summary.get("frame_count_analyzed", 0) or 0) <= 0:
        recommendations.append("No timing frames were available; do not use this run for threading decisions.")
    total_mean = _stat(timings, "total_ms", "mean")
    stage_names = (
        "detection_ms",
        "tracking_ms",
        "depth_fusion_ms",
        "segmentation_ms",
        "grasp_ms",
        "ranking_safety_ms",
        "recording_ms",
        "diagnostic_ms",
    )
    stage_means = {name: _stat(timings, name, "mean") for name in stage_names}
    stage_means = {name: value for name, value in stage_means.items() if value is not None}
    if total_mean and stage_means:
        dominant_name, dominant_ms = max(stage_means.items(), key=lambda item: item[1])
        share = dominant_ms / total_mean
        if share >= 0.35:
            recommendations.append(
                f"{dominant_name} is the dominant stage ({share:.0%} of median total latency); optimize or isolate this stage first."
            )

    frame_interval = float(summary.get("target_frame_interval_ms", 0.0) or 0.0)
    frame_age_p95 = _stat(timings, "frame_age_ms", "p95")
    if frame_interval > 0.0 and frame_age_p95 is not None and frame_age_p95 > frame_interval:
        recommendations.append(
            f"P95 frame age is {frame_age_p95:.1f} ms, above the {frame_interval:.1f} ms frame budget; use a bounded latest-frame queue."
        )

    dropped_metrics = [
        stats.get("max", 0.0)
        for name, stats in timings.items()
        if name.startswith("dropped_frames.") and isinstance(stats, Mapping)
    ]
    if any(float(value) > 0.0 for value in dropped_metrics):
        recommendations.append("Dropped frames were observed; compare frame age before increasing any queue capacity.")

    valid_target_rate = float(summary.get("valid_target_rate", 0.0) or 0.0)
    raw_detection_mean = _stat(timings, "workload.raw_detections", "mean")
    if valid_target_rate < 0.9 and raw_detection_mean is not None and raw_detection_mean < 0.9:
        recommendations.append(
            "Low target validity coincides with sparse detector output; inspect detector maximum confidence, "
            "above-threshold candidates, and active-target coast counts before tuning downstream stages."
        )

    queue_depths = {
        name.removeprefix("queue_depth."): stats
        for name, stats in timings.items()
        if name.startswith("queue_depth.") and isinstance(stats, Mapping)
    }
    for queue_name, stats in queue_depths.items():
        capacity = _stat(timings, f"queue_capacity.{queue_name}", "max")
        depth_p95 = float(stats.get("p95", 0.0))
        if capacity and depth_p95 >= 0.8 * capacity:
            recommendations.append(
                f"Queue {queue_name} is near capacity at P95; keep it bounded and inspect its consumer before adding workers."
            )

    tegrastats = summary.get("tegrastats", {})
    gpu_p95 = _stat(tegrastats, "gpu_util_percent", "p95")
    cpu_p95 = _stat(tegrastats, "cpu_mean_percent", "p95")
    if gpu_p95 is not None and gpu_p95 >= 85.0:
        recommendations.append("Jetson GPU utilization is saturated at P95; keep YOLO and GCNet serialized before testing GPU concurrency.")
    elif cpu_p95 is not None and cpu_p95 >= 85.0:
        recommendations.append("Jetson CPU utilization is high at P95; move recording and non-stateful postprocess work off the inference path.")

    if not queue_depths:
        recommendations.append(
            "No real queue metrics were recorded; implement bounded queues before using this run to choose queue capacities or worker counts."
        )
    if not tegrastats:
        recommendations.append("No tegrastats samples were available; rerun on Jetson for GPU, temperature, and power evidence.")
    if not recommendations:
        recommendations.append("No clear bottleneck threshold was crossed; retain the synchronous baseline and compare another representative scene.")
    return recommendations


def write_profile_reports(output_dir: str | Path, summary: Mapping[str, Any]) -> None:
    root = Path(output_dir)
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with (root / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("group", "metric", "count", "mean", "min", "p50", "p95", "p99", "max"))
        for group in ("timings", "process_resources", "tegrastats"):
            metrics = summary.get(group, {})
            if not isinstance(metrics, Mapping):
                continue
            for metric, stats in metrics.items():
                writer.writerow((group, metric, *(stats.get(name, "") for name in ("count", "mean", "min", "p50", "p95", "p99", "max"))))
    (root / "report.md").write_text(_render_markdown(summary), encoding="utf-8")


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Thread profiling report",
        "",
        f"- Frames analyzed: {summary.get('frame_count_analyzed', 0)} / {summary.get('frame_count_total', 0)}",
        f"- Warmup frames excluded: {summary.get('warmup_frames_excluded', 0)}",
        f"- Target FPS: {float(summary.get('target_fps', 0.0)):.2f}",
        f"- Effective pipeline FPS: {float(summary.get('effective_fps', 0.0)):.2f}",
        f"- End-to-end FPS including startup: {float(summary.get('collection_fps_including_startup', 0.0)):.2f}",
        f"- Valid target rate: {float(summary.get('valid_target_rate', 0.0)):.2%}",
        "",
        "## Timing summary",
        "",
        "| Metric | Mean | P50 | P95 | P99 | Max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric, stats in summary.get("timings", {}).items():
        if (metric.endswith("_ms") and metric not in _NON_METRIC_FIELDS) or metric.startswith("queue_wait_ms."):
            lines.append(
                f"| `{metric}` | {_fmt(stats, 'mean')} | {_fmt(stats, 'p50')} | {_fmt(stats, 'p95')} | {_fmt(stats, 'p99')} | {_fmt(stats, 'max')} |"
            )
    workload = {
        metric: stats
        for metric, stats in summary.get("timings", {}).items()
        if metric.startswith("workload.")
    }
    if workload:
        lines.extend(("", "## Workload summary", "", "| Metric | Mean | P50 | P95 | Max |", "|---|---:|---:|---:|---:|"))
        for metric, stats in workload.items():
            lines.append(
                f"| `{metric}` | {_fmt(stats, 'mean')} | {_fmt(stats, 'p50')} | {_fmt(stats, 'p95')} | {_fmt(stats, 'max')} |"
            )
    lines.extend(("", "## Recommendations", ""))
    for recommendation in summary.get("recommendations", []):
        lines.append(f"- {recommendation}")
    lines.extend(("", "Raw data and full resource statistics are available in the adjacent JSONL, JSON, and CSV files.", ""))
    return "\n".join(lines)


def _flatten_numeric(value: Mapping[str, Any], prefix: str = "") -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, item in value.items():
        if str(key) in _NON_METRIC_FIELDS:
            continue
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flattened.update(_flatten_numeric(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            flattened[name] = float(item)
    return flattened


def _stat(metrics: Any, name: str, statistic: str) -> float | None:
    if not isinstance(metrics, Mapping):
        return None
    value = metrics.get(name)
    if not isinstance(value, Mapping) or statistic not in value:
        return None
    return float(value[statistic])


def _fmt(stats: Mapping[str, Any], name: str) -> str:
    value = stats.get(name)
    return f"{float(value):.3f}" if value is not None else ""
