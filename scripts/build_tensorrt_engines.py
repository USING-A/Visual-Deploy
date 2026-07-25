"""Build device-specific TensorRT engines from the two deployment ONNX files."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    args = _parse_args()
    trtexec = args.trtexec or shutil.which("trtexec")
    if not trtexec:
        raise FileNotFoundError("trtexec was not found; pass --trtexec or install TensorRT on the Jetson")
    for source, output in (
        (args.yolo_onnx, args.yolo_engine),
        (args.gcnet_onnx, args.gcnet_engine),
    ):
        _build(Path(trtexec), source.resolve(), output.resolve(), fp16=not args.fp32)


def _build(trtexec: Path, source: Path, output: Path, *, fp16: bool) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"ONNX model not found: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [str(trtexec), f"--onnx={source}", f"--saveEngine={output}", "--skipInference"]
    if fp16:
        command.append("--fp16")
    subprocess.run(command, check=True)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError(f"TensorRT did not create a valid engine: {output}")
    print(f"Wrote TensorRT engine: {output}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run on the target Jetson; TensorRT engines are device specific.")
    parser.add_argument("--trtexec", type=Path, default=None)
    parser.add_argument("--yolo-onnx", type=Path, default=Path("weights/yolov10_sam_robust_standard.onnx"))
    parser.add_argument("--gcnet-onnx", type=Path, default=Path("weights/rgbd_gcnet_l03_robustft.onnx"))
    parser.add_argument("--yolo-engine", type=Path, default=Path("weights/yolov10_sam_robust_standard.engine"))
    parser.add_argument("--gcnet-engine", type=Path, default=Path("weights/rgbd_gcnet_l03_robustft.engine"))
    parser.add_argument("--fp32", action="store_true", help="disable the default FP16 engine build")
    return parser.parse_args()


if __name__ == "__main__":
    main()
