import pytest

from visual_deploy.inference import factory


def test_factory_propagates_tensorrt_buffer_reuse_switch(tmp_path, monkeypatch):
    config_path = tmp_path / "configs" / "deploy.yaml"
    weights_dir = tmp_path / "weights"
    config_path.parent.mkdir()
    weights_dir.mkdir()
    config_path.write_text("", encoding="utf-8")
    (weights_dir / "det.engine").write_bytes(b"engine")
    (weights_dir / "seg.engine").write_bytes(b"engine")
    captured = {}

    def fake_detector(weights, **kwargs):
        captured["detector"] = kwargs
        return object()

    def fake_segmentor(weights, **kwargs):
        captured["segmentor"] = kwargs
        return object()

    monkeypatch.setattr(factory, "YoloV10Detector", fake_detector)
    monkeypatch.setattr(factory, "GCNetSegmentor", fake_segmentor)
    config = {
        "detection": {"weights": "weights/det.engine", "reuse_buffers": False},
        "segmentation": {"weights": "weights/seg.engine", "reuse_buffers": False},
    }

    factory.build_detector(config_path, config)
    factory.build_segmentor(config_path, config)

    assert captured["detector"]["reuse_buffers"] is False
    assert captured["segmentor"]["reuse_buffers"] is False


def test_factory_rejects_non_boolean_buffer_reuse_value(tmp_path):
    config_path = tmp_path / "configs" / "deploy.yaml"
    weights_dir = tmp_path / "weights"
    config_path.parent.mkdir()
    weights_dir.mkdir()
    config_path.write_text("", encoding="utf-8")
    (weights_dir / "det.engine").write_bytes(b"engine")

    config = {"detection": {"weights": "weights/det.engine", "reuse_buffers": "false"}}

    with pytest.raises(ValueError, match="reuse_buffers must be a boolean"):
        factory.build_detector(config_path, config)
