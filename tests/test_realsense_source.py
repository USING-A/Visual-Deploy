import pytest

import visual_deploy.camera.realsense_source as realsense_source
from visual_deploy.camera.realsense_source import RealSenseSource, RealSenseUnavailableError
from scripts import run_realtime


def test_realsense_unavailable_error_is_importable():
    err = RealSenseUnavailableError("missing")
    assert "missing" in str(err)


def test_realsense_source_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setattr(realsense_source, "rs", None)
    monkeypatch.setattr(realsense_source, "_RS_IMPORT_ERROR", RuntimeError("dll load failed"))

    with pytest.raises(RealSenseUnavailableError, match="dll load failed") as exc_info:
        RealSenseSource()

    assert isinstance(exc_info.value.__cause__, RuntimeError)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"width": 0}, "width"),
        ({"height": 0}, "height"),
        ({"fps": 0}, "fps"),
        ({"width": 640.5}, "width"),
        ({"height": True}, "height"),
        ({"fps": "30"}, "fps"),
    ],
)
def test_realsense_source_validates_stream_config(monkeypatch, kwargs, message):
    monkeypatch.setattr(realsense_source, "rs", object())

    with pytest.raises(ValueError, match=message):
        RealSenseSource(**kwargs)


def test_realsense_source_requires_alignment_for_color_space_geometry(monkeypatch):
    monkeypatch.setattr(realsense_source, "rs", object())

    with pytest.raises(ValueError, match="align_to_color"):
        RealSenseSource(align_to_color=False)


def test_realtime_runner_passes_camera_config_without_truncation(monkeypatch):
    calls = []

    class FakeSource:
        def __init__(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(run_realtime, "RealSenseSource", FakeSource)

    run_realtime._build_source({"camera": {"width": 640.5, "height": True, "fps": "30"}})

    assert calls == [
        {
            "width": 640.5,
            "height": True,
            "fps": "30",
            "align_to_color": True,
            "spatial_filter": True,
            "temporal_filter": False,
            "hole_filling_filter": False,
        }
    ]
