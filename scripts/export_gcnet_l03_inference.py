"""Export a reparameterized RGBD-GCNet L03 inference TorchScript module.

The exported runtime is self-contained TorchScript. RGBD-GCNet, mmengine, and
mmsegmentation are only needed while running this export script.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

RGB_MEAN = (123.675, 116.28, 103.53)
RGB_STD = (58.395, 57.12, 57.375)


def _validate_threshold(threshold: float) -> float:
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
        raise ValueError(f"threshold must be finite and in [0, 1], got {threshold!r}")
    return threshold


def _as_batched_bgr(bgr: torch.Tensor) -> torch.Tensor:
    if bgr.ndim == 3:
        bgr = bgr.unsqueeze(0)
    elif bgr.ndim != 4:
        raise ValueError("BGR input must have shape (B,3,H,W) or (3,H,W)")
    if bgr.shape[1] != 3:
        raise ValueError("BGR input must have exactly 3 channels")
    return bgr


def _as_batched_depth(depth_mm: torch.Tensor) -> torch.Tensor:
    if depth_mm.ndim == 2:
        depth_mm = depth_mm.unsqueeze(0).unsqueeze(0)
    elif depth_mm.ndim == 3:
        depth_mm = depth_mm.unsqueeze(1)
    elif depth_mm.ndim != 4:
        raise ValueError("depth_mm must have shape (B,1,H,W), (B,H,W), or (H,W)")
    if depth_mm.shape[1] != 1:
        raise ValueError("depth_mm must have exactly 1 channel")
    return depth_mm


def normalize_bgr_depth(bgr: torch.Tensor, depth_mm: torch.Tensor) -> torch.Tensor:
    """Convert BGR pixels and millimeter depth into normalized RGB-D input.

    Accepts BGR image tensors with shape ``(B,3,H,W)`` or ``(3,H,W)`` and depth
    tensors with shape ``(B,1,H,W)``, ``(B,H,W)``, or ``(H,W)``. The RGB channels
    are ordered and normalized to match RGBD-GCNet's training preprocessor, while
    depth is converted from millimeters to meters.
    """
    bgr = _as_batched_bgr(bgr).to(dtype=torch.float32)
    depth_mm = _as_batched_depth(depth_mm).to(device=bgr.device, dtype=torch.float32)
    if bgr.shape[0] != depth_mm.shape[0]:
        raise ValueError(f"BGR and depth batch sizes must match, got {bgr.shape[0]} and {depth_mm.shape[0]}")
    if bgr.shape[-2:] != depth_mm.shape[-2:]:
        raise ValueError(f"BGR and depth spatial shapes must match, got {bgr.shape[-2:]} and {depth_mm.shape[-2:]}")

    rgb = bgr[:, [2, 1, 0], :, :]
    mean = torch.tensor(RGB_MEAN, device=rgb.device, dtype=torch.float64).view(1, 3, 1, 1)
    std = torch.tensor(RGB_STD, device=rgb.device, dtype=torch.float64).view(1, 3, 1, 1)
    rgb = ((rgb.to(dtype=torch.float64) - mean) / std).to(dtype=torch.float32)
    depth_m = depth_mm * 0.001
    return torch.cat([rgb, depth_m], dim=1)


def _decode_logits(model: nn.Module, normalized_rgbd: torch.Tensor) -> torch.Tensor:
    logits = model.decode_head(model.backbone(normalized_rgbd))
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"decode_head must return segmentation logits in eval mode, got {type(logits)!r}")
    return logits


def switch_backbone_to_deploy(model: nn.Module) -> None:
    """Switch the RGBD-GCNet backbone GCBlocks into deploy form."""
    backbone = getattr(model, "backbone", None)
    switch = getattr(backbone, "switch_to_deploy", None)
    if backbone is None or switch is None or not callable(switch):
        raise AttributeError("model.backbone.switch_to_deploy() is required for GCNet reparameterized export")
    switch()


def assert_deploy_equivalence(
    model: nn.Module,
    input_tensor: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> None:
    """Assert logits are stable when the backbone is switched to deploy mode.

    The same normalized RGB-D tensor is used before and after reparameterization.
    The improved segmentation-only head returns logits directly in eval mode, so
    this check intentionally does not unpack a historical ``(seg, geo)`` tuple.
    """
    model.eval()
    with torch.no_grad():
        before = _decode_logits(model, input_tensor)
        switch_backbone_to_deploy(model)
        model.eval()
        after = _decode_logits(model, input_tensor)
    if not torch.allclose(before, after, atol=atol, rtol=rtol):
        diff = torch.max(torch.abs(before - after)).item()
        raise AssertionError(f"deploy equivalence check failed: max abs diff {diff:.6g}")


class RGBDGCNetInferenceWrapper(nn.Module):
    """RGBD-GCNet inference wrapper for batched BGR image and millimeter depth.

    The TorchScript export supports only batched 4D inputs: BGR ``(B,3,H,W)`` and
    depth ``(B,1,H,W)``. Deployment loaders should add batch and channel
    dimensions before calling the traced module.

    Inside the RGBD-GCNet backbone, CME resizes ``raw_depth`` to ``depth_feat``
    size before confidence estimation. AFM then resizes ``depth_feat`` to the
    RGB/detail feature resolution for attention fusion.
    """

    def __init__(self, model: nn.Module, threshold: float = 0.5):
        super().__init__()
        self.backbone = model.backbone
        self.decode_head = model.decode_head
        self.threshold = _validate_threshold(threshold)
        self.register_buffer("rgb_mean", torch.tensor(RGB_MEAN, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("rgb_std", torch.tensor(RGB_STD, dtype=torch.float32).view(1, 3, 1, 1))

    def _normalize_batched(self, bgr: torch.Tensor, depth_mm: torch.Tensor) -> torch.Tensor:
        if not torch.jit.is_tracing():
            if bgr.ndim != 4:
                raise ValueError("BGR input must have batched shape (B,3,H,W)")
            if depth_mm.ndim != 4:
                raise ValueError("depth_mm must have batched shape (B,1,H,W)")
            if bgr.shape[1] != 3:
                raise ValueError("BGR input must have exactly 3 channels")
            if depth_mm.shape[1] != 1:
                raise ValueError("depth_mm must have exactly 1 channel")
            if bgr.shape[0] != depth_mm.shape[0]:
                raise ValueError(f"BGR and depth batch sizes must match, got {bgr.shape[0]} and {depth_mm.shape[0]}")
            if bgr.shape[-2:] != depth_mm.shape[-2:]:
                raise ValueError(f"BGR and depth spatial shapes must match, got {bgr.shape[-2:]} and {depth_mm.shape[-2:]}")

        rgb = bgr.to(dtype=torch.float32)[:, [2, 1, 0], :, :]
        depth_m = depth_mm.to(device=rgb.device, dtype=torch.float32) * 0.001
        return torch.cat([(rgb - self.rgb_mean) / self.rgb_std, depth_m], dim=1)

    def forward(self, bgr: torch.Tensor, depth_mm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        input_spatial = bgr.shape[-2:]
        normalized = self._normalize_batched(bgr, depth_mm)
        logits = self.decode_head(self.backbone(normalized))
        if not torch.jit.is_tracing() and not isinstance(logits, torch.Tensor):
            raise TypeError(f"decode_head must return segmentation logits in eval mode, got {type(logits)!r}")
        logits = F.interpolate(
            logits,
            size=input_spatial,
            mode="bilinear",
            align_corners=getattr(self.decode_head, "align_corners", False),
        )
        probs = torch.softmax(logits, dim=1)
        if not torch.jit.is_tracing() and probs.shape[1] < 2:
            raise ValueError(f"foreground probability requires at least 2 classes, got {probs.shape[1]}")
        foreground = probs[:, 1:2]
        mask = (foreground >= self.threshold).to(dtype=torch.uint8) * 255
        return foreground, mask


def assert_wrapper_parity(
    wrapper: nn.Module,
    traced: torch.jit.ScriptModule,
    bgr: torch.Tensor,
    depth_mm: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> None:
    """Assert eager and traced wrappers return the same foreground and mask."""
    wrapper.eval()
    traced.eval()
    with torch.no_grad():
        eager_foreground, eager_mask = wrapper(bgr, depth_mm)
        traced_foreground, traced_mask = traced(bgr, depth_mm)
    if not torch.allclose(eager_foreground, traced_foreground, atol=atol, rtol=rtol):
        diff = torch.max(torch.abs(eager_foreground - traced_foreground)).item()
        raise AssertionError(f"wrapper trace foreground parity failed: max abs diff {diff:.6g}")
    if not torch.equal(eager_mask, traced_mask):
        mismatch = torch.count_nonzero(eager_mask != traced_mask).item()
        raise AssertionError(f"wrapper trace mask parity failed: {mismatch} mismatched pixels")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="RGBD-GCNet mmengine config path")
    parser.add_argument("--checkpoint", required=True, help="trained RGBD-GCNet checkpoint path")
    parser.add_argument("--output", required=True, help="output TorchScript .pt path")
    parser.add_argument("--threshold", type=_validate_threshold, default=0.5, help="foreground probability threshold")
    parser.add_argument("--device", default="cpu", help="torch device for export, for example cpu or cuda:0")
    parser.add_argument("--rgbd-root", default=None, help="path to the RGBD-GCNet repository")
    return parser.parse_args(argv)


def _candidate_rgbd_roots(start: Path) -> list[Path]:
    candidates: list[Path] = []
    script_path = Path(__file__).resolve()
    anchors = [start.resolve(), *start.resolve().parents, script_path.parent, *script_path.parents]
    for anchor in anchors:
        candidates.append(anchor / "RGBD-GCNet")
        candidates.append(anchor.parent / "RGBD-GCNet")
    candidates.append(Path("D:/Github Code/RGBD-GCNet"))
    seen: set[Path] = set()
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def discover_rgbd_root(rgbd_root: str | None = None, start: Path | None = None) -> Path:
    if rgbd_root:
        root = Path(rgbd_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"--rgbd-root does not exist: {root}")
        return root

    start = Path.cwd() if start is None else start
    for candidate in _candidate_rgbd_roots(start):
        if (candidate / "apple_seg" / "models" / "rgbd_gcnet_cme_afm.py").is_file():
            return candidate
    searched = ", ".join(str(path) for path in _candidate_rgbd_roots(start)[:8])
    raise FileNotFoundError(f"Could not discover RGBD-GCNet root. Use --rgbd-root. Searched: {searched}")


def setup_rgbd_root(rgbd_root: Path) -> None:
    if os.name == "nt":
        local_appdata = rgbd_root / ".tmp" / "localappdata"
        local_appdata.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("WIN_PD_OVERRIDE_LOCAL_APPDATA", str(local_appdata))

    for path in (rgbd_root, rgbd_root / "GCNet" / "mmsegmentation"):
        path_str = str(path)
        if path.is_dir() and path_str not in sys.path:
            sys.path.insert(0, path_str)


def register_rgbd_modules() -> None:
    import apple_seg.models  # noqa: F401


def make_deterministic_export_inputs(
    device: str | torch.device,
    height: int = 256,
    width: int = 256,
) -> tuple[torch.Tensor, torch.Tensor]:
    values = height * width
    b = torch.linspace(0.0, 255.0, steps=values, device=device, dtype=torch.float32).reshape(1, 1, height, width)
    g = torch.linspace(255.0, 0.0, steps=values, device=device, dtype=torch.float32).reshape(1, 1, height, width)
    r = torch.linspace(32.0, 224.0, steps=values, device=device, dtype=torch.float32).reshape(1, 1, height, width)
    bgr = torch.cat([b, g, r], dim=1)
    depth_mm = torch.linspace(250.0, 750.0, steps=values, device=device, dtype=torch.float32).reshape(1, 1, height, width)
    return bgr, depth_mm


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rgbd_root = discover_rgbd_root(args.rgbd_root)
    setup_rgbd_root(rgbd_root)

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmengine.runner import Runner

    register_rgbd_modules()

    cfg = Config.fromfile(args.config)
    init_default_scope(cfg.get("default_scope", "mmseg"))
    cfg.load_from = args.checkpoint
    if "work_dir" not in cfg:
        cfg.work_dir = str(Path(args.checkpoint).resolve().parent)

    runner = Runner.from_cfg(cfg)
    runner.load_checkpoint(args.checkpoint)
    model = runner.model.to(args.device)
    model.eval()

    bgr, depth_mm = make_deterministic_export_inputs(args.device)
    normalized = normalize_bgr_depth(bgr, depth_mm)
    assert_deploy_equivalence(model, normalized)

    wrapper = RGBDGCNetInferenceWrapper(model, threshold=args.threshold).to(args.device)
    wrapper.eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (bgr, depth_mm), strict=False)
    assert_wrapper_parity(wrapper, traced, bgr, depth_mm)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(traced, str(output))
    print(f"Wrote reparameterized RGBD-GCNet inference TorchScript to {output}")


if __name__ == "__main__":
    main()
