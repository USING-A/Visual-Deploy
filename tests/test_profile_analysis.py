import json

import pytest

from visual_deploy.observability.profile_analysis import (
    build_profile_summary,
    parse_tegrastats_line,
    percentile,
    write_profile_reports,
)


def test_parse_tegrastats_line_handles_orin_multi_gpc_and_power_rails():
    line = (
        "RAM 4601/63895MB (lfb 3x64MB) "
        "CPU [10%@729,20%@729,off,30%@729] GR3D_FREQ 99%@[1098,1098] "
        "cpu@45.062C gpu@50C VDD_GPU_SOC 3065mW/3000mW "
        "VIN_SYS_5V0 3623mW/3600mW"
    )

    sample = parse_tegrastats_line(line)

    assert sample["system_ram_used_mb"] == 4601.0
    assert sample["system_ram_total_mb"] == 63895.0
    assert sample["cpu_mean_percent"] == 20.0
    assert sample["cpu_max_core_percent"] == 30.0
    assert sample["cpu_active_cores"] == 3.0
    assert sample["gpu_util_percent"] == 99.0
    assert sample["temperature_max_c"] == 50.0
    assert sample["power_vdd_gpu_soc_mw"] == 3065.0
    assert sample["input_power_mw"] == 3623.0
    assert sample["input_power_average_mw"] == 3600.0


def test_percentile_uses_linear_interpolation():
    assert percentile([0.0, 10.0], 95.0) == pytest.approx(9.5)
    with pytest.raises(ValueError):
        percentile([], 50.0)


def test_build_profile_summary_excludes_warmup_and_preserves_nested_queue_metrics():
    timings = [
        {"frame_id": 0, "valid_target": False, "total_ms": 100.0, "detection_ms": 90.0},
        {
            "frame_id": 1,
            "valid_target": True,
            "total_ms": 20.0,
            "detection_ms": 10.0,
            "frame_age_ms": 40.0,
            "queue_depth": {"capture_to_inference": 2},
            "queue_capacity": {"capture_to_inference": 2},
            "dropped_frames": {"capture": 1},
            "completed_monotonic_ms": 1000.0,
        },
        {
            "frame_id": 2,
            "valid_target": False,
            "total_ms": 30.0,
            "detection_ms": 20.0,
            "frame_age_ms": 50.0,
            "queue_depth": {"capture_to_inference": 1},
            "queue_capacity": {"capture_to_inference": 2},
            "dropped_frames": {"capture": 1},
            "completed_monotonic_ms": 1050.0,
        },
    ]
    tegra = [{"gpu_util_percent": 90.0}, {"gpu_util_percent": 95.0}]

    summary = build_profile_summary(timings, [], tegra, warmup_frames=1, target_fps=30.0)

    assert summary["frame_count_analyzed"] == 2
    assert summary["valid_target_rate"] == 0.5
    assert summary["timings"]["total_ms"]["mean"] == 25.0
    assert summary["timings"]["queue_depth.capture_to_inference"]["max"] == 2.0
    assert summary["effective_fps"] == 20.0
    assert "completed_monotonic_ms" not in summary["timings"]
    assert any("bounded latest-frame queue" in item for item in summary["recommendations"])
    assert any("GPU utilization" in item for item in summary["recommendations"])


def test_write_profile_reports_creates_machine_and_human_readable_outputs(tmp_path):
    summary = build_profile_summary(
        [{"total_ms": 10.0, "detection_ms": 5.0, "valid_target": True}],
        [{"process_cpu_percent": 20.0, "process_rss_mb": 100.0}],
        [],
        target_fps=30.0,
    )
    summary["effective_fps"] = 25.0

    write_profile_reports(tmp_path, summary)

    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["frame_count_analyzed"] == 1
    assert "timings,total_ms" in (tmp_path / "summary.csv").read_text(encoding="utf-8")
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Effective pipeline FPS: 25.00" in report
    assert "## Recommendations" in report


def test_empty_profile_is_explicitly_not_actionable():
    summary = build_profile_summary([], [], [], warmup_frames=0)
    assert any("No timing frames" in item for item in summary["recommendations"])
