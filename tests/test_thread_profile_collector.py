import argparse
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "collect_thread_profile.py"
SPEC = importlib.util.spec_from_file_location("collect_thread_profile", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)


def test_profile_config_absolutizes_models_and_enables_only_timing(tmp_path):
    config_path = tmp_path / "repo" / "configs" / "deploy.yaml"
    config_path.parent.mkdir(parents=True)
    session_dir = tmp_path / "profile"
    config = {
        "detection": {"weights": "weights/yolo.onnx"},
        "segmentation": {"weights": "weights/gcnet.onnx"},
        "debug": {"enabled": True},
        "diagnostics": {"enabled": True},
        "recording": {"enabled": False, "save_frames": True},
    }

    generated = COLLECTOR._build_profile_config(config, config_path, session_dir)

    assert Path(generated["detection"]["weights"]).is_absolute()
    assert Path(generated["segmentation"]["weights"]).is_absolute()
    assert generated["profiling"] == {"enabled": True}
    assert generated["debug"] == {"enabled": False}
    assert generated["diagnostics"] == {"enabled": False}
    assert generated["recording"]["enabled"] is True
    assert generated["recording"]["save_timings"] is True
    assert generated["recording"]["save_frames"] is False


def test_build_offline_command_requires_all_rgbd_arguments(tmp_path):
    args = argparse.Namespace(
        mode="offline",
        python="python",
        frames=10,
        rgb=None,
        depth=None,
        fx=None,
        fy=None,
        ppx=None,
        ppy=None,
        use_mock_models=False,
    )
    with pytest.raises(ValueError, match="--rgb"):
        COLLECTOR._build_child_command(args, tmp_path / "profile.yaml")


def test_build_realtime_command_supports_synchronous_comparison(tmp_path):
    args = argparse.Namespace(mode="realtime", python="python", frames=30, sync=True)

    command = COLLECTOR._build_child_command(args, tmp_path / "profile.yaml")

    assert command[-1] == "--sync"
    assert "run_realtime.py" in command[1]


def test_tegrastats_resolution_supports_off_and_explicit_executable(tmp_path):
    assert COLLECTOR._resolve_tegrastats("off", 250) is None
    executable = tmp_path / "tegrastats"
    executable.write_text("", encoding="utf-8")
    assert COLLECTOR._resolve_tegrastats(str(executable), 250) == [str(executable), "--interval", "250"]
    with pytest.raises(FileNotFoundError):
        COLLECTOR._resolve_tegrastats(str(tmp_path / "missing"), 250)


def test_offline_collection_runs_child_and_generates_reports(tmp_path):
    rgb_path = tmp_path / "rgb.png"
    depth_path = tmp_path / "depth.npy"
    assert cv2.imwrite(str(rgb_path), np.zeros((480, 640, 3), dtype=np.uint8))
    np.save(depth_path, np.full((480, 640), 1000.0, dtype=np.float32))
    config_path = tmp_path / "deploy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "camera": {"fps": 30},
                "detection": {"weights": "missing_yolo.onnx"},
                "segmentation": {"weights": "missing_gcnet.onnx"},
                "tracking": {"high_conf_threshold": 0.5, "low_conf_threshold": 0.1, "min_hits": 1},
                "recording": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        mode="offline",
        config=str(config_path),
        frames=3,
        warmup_frames=1,
        sample_interval_ms=10,
        timeout_s=30.0,
        output_root=str(tmp_path / "profiles"),
        session_name="integration",
        python=sys.executable,
        tegrastats="off",
        rgb=rgb_path,
        depth=depth_path,
        fx=600.0,
        fy=600.0,
        ppx=320.0,
        ppy=240.0,
        use_mock_models=True,
    )

    result = COLLECTOR.collect_profile(args)

    session_dir = Path(result["session_dir"])
    assert result["status"] == "complete"
    assert result["exit_code"] == 0
    assert result["frame_count"] == 3
    summary = json.loads((session_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["frame_count_analyzed"] == 2
    assert summary["timings"]["total_ms"]["count"] == 2.0
    assert (session_dir / "process_resources.jsonl").is_file()
    assert (session_dir / "report.md").is_file()


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"timed_out": True}, ("timed_out", 124)),
        ({"interrupted": True}, ("interrupted", 130)),
        ({"child_return_code": 7}, ("child_failed", 7)),
        ({"frame_count": 0}, ("invalid_output", 2)),
        ({"analyzed_frames": 0}, ("invalid_output", 2)),
        ({"tegrastats_sample_count": 0}, ("incomplete_telemetry", 3)),
        ({"tegrastats_exited_early": True}, ("incomplete_telemetry", 3)),
    ],
)
def test_profile_status_fails_closed_for_incomplete_runs(overrides, expected):
    values = {
        "child_return_code": 0,
        "timed_out": False,
        "interrupted": False,
        "frame_count": 10,
        "expected_frames": 10,
        "analyzed_frames": 9,
        "tegrastats_requested": True,
        "tegrastats_sample_count": 2,
        "tegrastats_exited_early": False,
    }
    values.update(overrides)
    assert COLLECTOR._determine_profile_status(**values) == expected


def test_collection_args_reject_warmup_that_consumes_the_run():
    args = argparse.Namespace(
        mode="realtime", frames=10, warmup_frames=10, sample_interval_ms=1000, timeout_s=None, sync=False
    )
    with pytest.raises(ValueError, match="smaller than frames"):
        COLLECTOR._validate_collection_args(args)


def test_collection_args_reject_sync_for_offline_mode():
    args = argparse.Namespace(
        mode="offline", frames=10, warmup_frames=1, sample_interval_ms=1000, timeout_s=None, sync=True
    )
    with pytest.raises(ValueError, match="realtime"):
        COLLECTOR._validate_collection_args(args)
