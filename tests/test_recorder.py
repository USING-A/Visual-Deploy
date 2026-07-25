import json

import numpy as np
import pytest

from visual_deploy.recording.recorder import RunRecorder


def test_recorder_writes_jsonl_records(tmp_path):
    rec = RunRecorder(tmp_path, run_name="test_run", config={"a": 1})
    rec.write_detection({"frame_id": 1, "detections": []})
    rec.write_candidate({"frame_id": 1, "track_id": 2})
    rec.write_target({"frame_id": 1, "valid": False, "reason": "no_valid_grasp_candidate"})

    run_dir = tmp_path / "test_run"
    assert (run_dir / "run_config.yaml").exists()
    det = json.loads((run_dir / "detections.jsonl").read_text(encoding="utf-8").splitlines()[0])
    target = json.loads((run_dir / "targets.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert det["frame_id"] == 1
    assert target["reason"] == "no_valid_grasp_candidate"


def test_recorder_creates_artifact_directories(tmp_path):
    rec = RunRecorder(tmp_path, run_name="test_run")

    assert (rec.run_dir / "frames").is_dir()
    assert (rec.run_dir / "masks").is_dir()
    assert (rec.run_dir / "overlays").is_dir()
    assert (rec.run_dir / "depth").is_dir()


def test_recorder_writes_timing_and_event_artifacts(tmp_path):
    rec = RunRecorder(tmp_path, run_name="test_run")
    color = np.zeros((12, 16, 3), dtype=np.uint8)
    depth = np.full((12, 16), 250.0, dtype=np.float32)
    mask = np.zeros((12, 16), dtype=bool)
    mask[3:9, 4:12] = True
    artifacts = rec.save_event_artifacts(
        7,
        color_bgr=color,
        depth_mm=depth,
        masks=[(3, mask)],
        overlay_bgr=color,
    )
    rec.write_timing({"frame_id": 7, "total_ms": 10.0})
    rec.write_event({"frame_id": 7, "artifacts": artifacts})

    run_dir = tmp_path / "test_run"
    assert (run_dir / artifacts["frame"]).is_file()
    assert (run_dir / artifacts["depth"]).is_file()
    assert (run_dir / artifacts["masks"][0]).is_file()
    assert (run_dir / artifacts["overlay"]).is_file()
    assert json.loads((run_dir / "timings.jsonl").read_text(encoding="utf-8"))["total_ms"] == 10.0
    assert json.loads((run_dir / "events.jsonl").read_text(encoding="utf-8"))["frame_id"] == 7


def test_recorder_writes_error_records(tmp_path):
    rec = RunRecorder(tmp_path, run_name="test_run")
    rec.write_error({"frame_id": 3, "error": "failed"})

    line = (tmp_path / "test_run" / "errors.jsonl").read_text(encoding="utf-8").strip()
    assert json.loads(line) == {"frame_id": 3, "error": "failed"}


def test_recorder_rejects_non_mapping_record(tmp_path):
    rec = RunRecorder(tmp_path, run_name="test_run")

    with pytest.raises(TypeError, match="record"):
        rec.write_target(["not", "a", "mapping"])  # type: ignore[arg-type]


@pytest.mark.parametrize("run_name", ["../outside", "..\\outside", "C:\\outside", "", ".", ".."])
def test_recorder_rejects_unsafe_run_name(tmp_path, run_name):
    with pytest.raises(ValueError, match="run_name"):
        RunRecorder(tmp_path, run_name=run_name)


def test_recorder_default_run_names_do_not_collide(tmp_path):
    first = RunRecorder(tmp_path)
    second = RunRecorder(tmp_path)

    assert first.run_dir != second.run_dir
    assert first.run_dir.exists()
    assert second.run_dir.exists()


def test_recorder_rejects_existing_run_name(tmp_path):
    RunRecorder(tmp_path, run_name="test_run")

    with pytest.raises(FileExistsError):
        RunRecorder(tmp_path, run_name="test_run")


def test_recorder_master_switch_creates_no_files(tmp_path):
    rec = RunRecorder(tmp_path, run_name="disabled", enabled=False)
    rec.write_detection({"frame_id": 1})
    rec.write_timing({"frame_id": 1})
    assert not rec.run_dir.exists()


def test_recorder_channel_switch_limits_debug_outputs(tmp_path):
    rec = RunRecorder(tmp_path, run_name="targets_only", channels={"targets"})
    rec.write_detection({"frame_id": 1})
    rec.write_target({"frame_id": 1})
    assert not (rec.run_dir / "detections.jsonl").exists()
    assert (rec.run_dir / "targets.jsonl").exists()
