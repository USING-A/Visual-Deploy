"""Export the reparameterized RGBD-GCNet TorchScript artifact to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    args = _parse_args()
    source = args.torchscript.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    model = torch.jit.load(str(source), map_location="cpu").eval()
    bgr = torch.zeros((1, 3, 256, 256), dtype=torch.float32)
    depth = torch.full((1, 1, 256, 256), args.verify_depth_mm, dtype=torch.float32)
    with torch.no_grad():
        reference_prob, _ = model(bgr, depth)

    with tempfile.TemporaryDirectory(prefix="gcnet_export_", dir=output.parent) as temp_dir:
        full = Path(temp_dir) / "gcnet_full.onnx"
        torch.onnx.export(
            model,
            (bgr, depth),
            str(full),
            opset_version=args.opset,
            input_names=["bgr", "depth_mm"],
            output_names=["prob", "mask"],
            do_constant_folding=True,
        )
        import onnx

        onnx.utils.extract_model(str(full), str(output), ["bgr", "depth_mm"], ["prob"], check_model=True)

    max_error, mean_error = _verify_onnx(output, bgr.numpy(), depth.numpy(), reference_prob.numpy())
    metadata = {
        "format": "rgbd-gcnet-reparameterized-probability-onnx",
        "source_torchscript": str(source),
        "source_torchscript_sha256": _sha256(source),
        "onnx_sha256": _sha256(output),
        "opset": args.opset,
        "inputs": {"bgr": [1, 3, 256, 256], "depth_mm": [1, 1, 256, 256]},
        "output": {"prob": [1, 1, 256, 256]},
        "mask_postprocess": "prob >= segmentation.threshold",
        "verification_mean_abs_error": mean_error,
        "verification_max_abs_error": max_error,
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote RGBD-GCNet ONNX: {output}")


def _verify_onnx(path: Path, bgr: np.ndarray, depth: np.ndarray, reference: np.ndarray) -> tuple[float, float]:
    import onnx
    import onnxruntime as ort

    model = onnx.load(path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    actual = session.run(["prob"], {"bgr": bgr, "depth_mm": depth})[0]
    delta = np.abs(actual - reference)
    max_error, mean_error = float(delta.max()), float(delta.mean())
    if actual.shape != (1, 1, 256, 256) or not np.isfinite(actual).all() or max_error > 1e-3:
        raise RuntimeError(f"GCNet ONNX verification failed: shape={actual.shape}, max_abs_error={max_error}")
    return max_error, mean_error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torchscript", default="weights/rgbd_gcnet_l03_robustft_inference.pt", type=Path)
    parser.add_argument("--output", default="weights/rgbd_gcnet_l03_robustft.onnx", type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--verify-depth-mm", type=float, default=500.0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
