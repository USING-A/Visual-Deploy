from pathlib import Path

import numpy as np
import pytest

from visual_deploy.config import load_config, resolve_path
from visual_deploy.types import CameraIntrinsics, DeployFrame, Detection


def test_load_default_config_has_runtime_sections():
    cfg = load_config(Path("configs/deploy.yaml"))
    assert set(cfg) >= {
        "camera",
        "detection",
        "tracking",
        "roi",
        "segmentation",
        "depth_fusion",
        "grasp",
        "ranking",
        "safety",
        "profiling",
        "debug",
        "diagnostics",
        "recording",
    }
    assert cfg["roi"]["size"] == 256
    assert cfg["roi"]["pad_ratio"] == 0.20
    assert cfg["camera"]["align_to_color"] is True
    assert cfg["detection"]["backend"] == "onnxruntime"
    assert cfg["detection"]["weights"].endswith(".onnx")
    assert cfg["segmentation"]["weights"].endswith(".onnx")
    assert cfg["tracking"]["high_conf_threshold"] == 0.50
    assert cfg["tracking"]["low_conf_threshold"] == 0.10
    assert cfg["tracking"]["min_hits"] == 1
    assert cfg["segmentation"]["depth_input_unit"] == "mm"
    assert cfg["safety"]["enabled"] is True
    assert cfg["grasp"]["candidate_top_k"] == 256
    assert cfg["grasp"]["exhaustive_fallback"] is True
    assert cfg["grasp"]["shadow_verify_every_n_frames"] == 0


def test_cpu_config_preserves_production_safety_and_temporal_contracts():
    production = load_config(Path("configs/deploy.yaml"))
    cpu = load_config(Path("configs/deploy.cpu.local.yaml"))

    assert cpu["tracking"] == production["tracking"]
    assert cpu["depth_fusion"] == production["depth_fusion"]
    assert cpu["grasp"] == production["grasp"]
    assert cpu["safety"] == production["safety"]
    assert cpu["profiling"] == production["profiling"]
    assert cpu["debug"] == production["debug"]
    assert cpu["diagnostics"] == production["diagnostics"]


def test_load_config_rejects_non_mapping_yaml(tmp_path):
    cfg_path = tmp_path / "deploy.yaml"
    cfg_path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(cfg_path)


def test_resolve_path_is_relative_to_project_root(tmp_path):
    cfg_path = tmp_path / "configs" / "deploy.yaml"
    cfg_path.parent.mkdir()
    cfg_path.write_text("weights: weights/model.pt\n", encoding="utf-8")
    assert resolve_path(cfg_path, "weights/model.pt") == tmp_path / "weights" / "model.pt"


def test_resolve_path_preserves_absolute_paths(tmp_path):
    cfg_path = tmp_path / "configs" / "deploy.yaml"
    absolute_path = tmp_path / "weights" / "model.pt"
    assert resolve_path(cfg_path, absolute_path) == absolute_path


def test_resolve_path_for_non_config_file_is_relative_to_file_parent(tmp_path):
    cfg_path = tmp_path / "runtime" / "deploy.yaml"
    cfg_path.parent.mkdir()
    assert resolve_path(cfg_path, "weights/model.pt") == cfg_path.parent / "weights" / "model.pt"


def test_shared_types_are_plain_dataclasses():
    intr = CameraIntrinsics(fx=600.0, fy=601.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    det = Detection(bbox_xyxy=(1.0, 2.0, 3.0, 4.0), class_id=0, confidence=0.9, label="apple")
    assert intr.fx == 600.0
    assert det.weighted_confidence == 0.9


def test_deploy_frame_equality_is_identity_based_for_array_fields():
    intr = CameraIntrinsics(fx=600.0, fy=601.0, ppx=320.0, ppy=240.0, depth_scale=0.001)
    color = np.zeros((2, 2, 3), dtype=np.uint8)
    depth = np.zeros((2, 2), dtype=np.uint16)
    frame_a = DeployFrame(
        frame_id=1,
        timestamp_ms=10.0,
        color_bgr=color,
        depth_mm=depth,
        intrinsics=intr,
    )
    frame_b = DeployFrame(
        frame_id=1,
        timestamp_ms=10.0,
        color_bgr=color.copy(),
        depth_mm=depth.copy(),
        intrinsics=intr,
    )

    assert frame_a == frame_a
    assert frame_a != frame_b
