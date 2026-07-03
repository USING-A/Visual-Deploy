import json

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
