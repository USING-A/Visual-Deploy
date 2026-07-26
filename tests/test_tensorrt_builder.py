import importlib.util
import subprocess
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "build_tensorrt_engines.py"
SPEC = importlib.util.spec_from_file_location("build_tensorrt_engines", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def _engine_path(command: list[str]) -> Path:
    return Path(next(value.split("=", 1)[1] for value in command if value.startswith("--saveEngine=")))


def test_tensorrt_build_atomically_replaces_final_engine(tmp_path, monkeypatch):
    source = tmp_path / "model.onnx"
    output = tmp_path / "model.engine"
    source.write_bytes(b"onnx")
    output.write_bytes(b"old")

    def fake_run(command, check):
        assert check is True
        assert "--fp16" in command
        temporary = _engine_path(command)
        assert temporary != output
        temporary.write_bytes(b"new-engine")

    monkeypatch.setattr(BUILDER.subprocess, "run", fake_run)
    BUILDER._build(tmp_path / "trtexec", source, output, fp16=True)

    assert output.read_bytes() == b"new-engine"
    assert not list(tmp_path.glob("*.tmp"))


def test_tensorrt_build_failure_preserves_previous_engine(tmp_path, monkeypatch):
    source = tmp_path / "model.onnx"
    output = tmp_path / "model.engine"
    source.write_bytes(b"onnx")
    output.write_bytes(b"known-good")

    def fake_run(command, check):
        _engine_path(command).write_bytes(b"partial")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(BUILDER.subprocess, "run", fake_run)
    with pytest.raises(subprocess.CalledProcessError):
        BUILDER._build(tmp_path / "trtexec", source, output, fp16=False)

    assert output.read_bytes() == b"known-good"
    assert not list(tmp_path.glob("*.tmp"))
