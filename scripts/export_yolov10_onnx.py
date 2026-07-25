"""Export a SAM-supervised checkpoint as a standard YOLOv10 ONNX graph."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import torch


def main() -> None:
    args = _parse_args()
    source_repo = args.source_repo.resolve()
    sys.path.insert(0, str(source_repo))
    from ultralytics import YOLOv10

    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("model", "ema"):
        model = payload.get(key)
        if model is not None:
            _strip_training_only_modules(model)

    with tempfile.TemporaryDirectory(prefix="yolov10_export_", dir=output.parent) as temp_dir:
        deploy_pt = Path(temp_dir) / "yolov10_sam_robust_standard.pt"
        torch.save(payload, deploy_pt)
        loaded = YOLOv10(str(deploy_pt))
        _assert_standard_runtime_graph(loaded.model)
        exported = Path(
            loaded.export(
                format="onnx",
                imgsz=args.image_size,
                batch=1,
                opset=args.opset,
                simplify=args.simplify,
                dynamic=False,
                half=False,
                device="cpu",
            )
        )
        exported.replace(output)

    _verify_onnx(output, args.image_size)
    metadata = {
        "format": "standard-yolov10-end2end-onnx",
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _sha256(checkpoint),
        "onnx_sha256": _sha256(output),
        "image_size": args.image_size,
        "opset": args.opset,
        "input": {"images": [1, 3, args.image_size, args.image_size]},
        "output": {"output0": [1, 300, 6]},
        "class_names": {"0": "apple"},
        "training_only_modules_removed": ["SAMROIMaskDistiller", "sam_roi_distill metadata"],
        "eca_modules_present": False,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote standard YOLOv10 ONNX: {output}")


def _strip_training_only_modules(model: torch.nn.Module) -> None:
    if hasattr(model, "sam_roi_distiller"):
        delattr(model, "sam_roi_distiller")
    yaml = getattr(model, "yaml", None)
    if isinstance(yaml, dict):
        yaml.pop("sam_roi_distill", None)
        yaml["yaml_file"] = "yolov10n.yaml"
    head = model.model[-1]
    if hasattr(head, "capture_distill_features"):
        head.capture_distill_features = False
    if hasattr(head, "distill_features"):
        delattr(head, "distill_features")


def _assert_standard_runtime_graph(model: torch.nn.Module) -> None:
    module_names = {type(module).__name__ for module in model.modules()}
    forbidden = sorted(name for name in module_names if "eca" in name.lower() or name == "SAMROIMaskDistiller")
    if forbidden or hasattr(model, "sam_roi_distiller"):
        raise RuntimeError(f"training/custom modules remain in YOLOv10 graph: {forbidden}")
    if type(model).__name__ != "YOLOv10DetectionModel" or type(model.model[-1]).__name__ != "v10Detect":
        raise RuntimeError("checkpoint is not a standard YOLOv10 detection model")


def _verify_onnx(path: Path, image_size: int) -> None:
    import numpy as np
    import onnx

    model = onnx.load(path)
    onnx.checker.check_model(model)
    try:
        import onnxruntime as ort
    except ImportError:
        return
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    output = session.run(None, {"images": np.zeros((1, 3, image_size, image_size), dtype=np.float32)})[0]
    if output.shape != (1, 300, 6) or not np.isfinite(output).all():
        raise RuntimeError(f"unexpected YOLOv10 ONNX output: {output.shape}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", default="weights/yolov10_sam_robust_standard.onnx", type=Path)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--simplify", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
