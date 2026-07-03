from __future__ import annotations

from dataclasses import dataclass

from visual_deploy.types import BBoxXYXY, CameraIntrinsics


def square_pad_bbox(bbox: BBoxXYXY, pad_ratio: float, img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    if img_w <= 0 or img_h <= 0:
        raise ValueError("image dimensions must be positive")
    if pad_ratio < 0:
        raise ValueError("pad_ratio must be non-negative")

    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        raise ValueError("bbox must have positive width and height")

    center_x = (x0 + x1) * 0.5
    center_y = (y0 + y1) * 0.5
    side = max(width, height) * (1.0 + pad_ratio)
    half_side = side * 0.5

    crop_x0 = max(0, int(round(center_x - half_side)))
    crop_y0 = max(0, int(round(center_y - half_side)))
    crop_x1 = min(img_w, int(round(center_x + half_side)))
    crop_y1 = min(img_h, int(round(center_y + half_side)))
    if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
        raise ValueError("padded bbox produced an empty crop")

    return crop_x0, crop_y0, crop_x1, crop_y1


@dataclass(frozen=True)
class RoiTransform:
    crop_xyxy: tuple[int, int, int, int]
    roi_size: int = 256

    def __post_init__(self) -> None:
        x0, y0, x1, y1 = self.crop_xyxy
        if self.roi_size <= 0:
            raise ValueError("roi_size must be positive")
        if x1 <= x0 or y1 <= y0:
            raise ValueError("crop_xyxy must have positive width and height")

    @classmethod
    def from_bbox(
        cls,
        bbox: BBoxXYXY,
        pad_ratio: float,
        img_w: int,
        img_h: int,
        roi_size: int = 256,
    ) -> RoiTransform:
        return cls(crop_xyxy=square_pad_bbox(bbox, pad_ratio, img_w, img_h), roi_size=roi_size)

    @property
    def crop_width(self) -> int:
        x0, _, x1, _ = self.crop_xyxy
        return x1 - x0

    @property
    def crop_height(self) -> int:
        _, y0, _, y1 = self.crop_xyxy
        return y1 - y0

    @property
    def scale_x(self) -> float:
        return self.roi_size / self.crop_width

    @property
    def scale_y(self) -> float:
        return self.roi_size / self.crop_height

    def image_to_roi(self, u: float, v: float) -> tuple[float, float]:
        x0, y0, _, _ = self.crop_xyxy
        return (u - x0) * self.scale_x, (v - y0) * self.scale_y

    def roi_to_image(self, u: float, v: float) -> tuple[float, float]:
        x0, y0, _, _ = self.crop_xyxy
        return (u / self.scale_x) + x0, (v / self.scale_y) + y0

    def adjust_intrinsics(self, intr: CameraIntrinsics) -> CameraIntrinsics:
        x0, y0, _, _ = self.crop_xyxy
        return CameraIntrinsics(
            fx=intr.fx * self.scale_x,
            fy=intr.fy * self.scale_y,
            ppx=(intr.ppx - x0) * self.scale_x,
            ppy=(intr.ppy - y0) * self.scale_y,
            depth_scale=intr.depth_scale,
        )
