"""Deterministic image-domain processing; never process labels differently."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from io import BytesIO
import math
from PIL import Image, ImageFilter


@dataclass(frozen=True)
class Condition:
    """One fixed processing setting; severity ranks settings within its family only."""

    name: str
    family: str
    severity: float = 0.0
    factor: float | None = None
    quality: int | None = None
    radius: float | None = None
    percent: int | None = None

    def to_dict(self):
        return asdict(self)


CONDITIONS = [
    Condition("clean", "clean"),
    Condition("jpeg_q95", "jpeg", 1, quality=95),
    Condition("jpeg_q75", "jpeg", 2, quality=75),
    Condition("jpeg_q50", "jpeg", 3, quality=50),
    Condition("resize_075", "resize", 1, factor=0.75),
    Condition("resize_050", "resize", 2, factor=0.50),
    Condition("resize_025", "resize", 3, factor=0.25),
    Condition("crop_area075", "crop", 1, factor=0.75),
    Condition("crop_area050", "crop", 2, factor=0.50),
    Condition("crop_area025", "crop", 3, factor=0.25),
    Condition("blur_sigma05", "blur", 1, radius=0.5),
    Condition("blur_sigma10", "blur", 2, radius=1.0),
    Condition("blur_sigma20", "blur", 3, radius=2.0),
    Condition("sharpen_050", "sharpen", 1, radius=1.0, percent=50),
    Condition("sharpen_100", "sharpen", 2, radius=1.0, percent=100),
    Condition("sharpen_200", "sharpen", 3, radius=1.0, percent=200),
    Condition("resize075_jpeg75", "resize_jpeg", 1, factor=0.75, quality=75),
    Condition("resize050_jpeg75", "resize_jpeg", 2, factor=0.50, quality=75),
    Condition("resize050_jpeg50", "resize_jpeg", 3, factor=0.50, quality=50),
]
CONDITION_MAP = {item.name: item for item in CONDITIONS}


def apply_condition(image: Image.Image, condition: Condition) -> Image.Image:
    """RGB→processing→RGB, then FRE performs the released grayscale conversion.

    Resize is one antialiased Lanczos downsample (no resize-back). Crop preserves
    the stated image area, centered, with no resampling. JPEG uses 4:2:0 chroma
    subsampling, baseline coding, and an in-memory encode/decode. Blur uses
    Pillow GaussianBlur, whose radius is the standard deviation. Sharpen uses
    UnsharpMask(radius=1, percent=..., threshold=0). Compound order is resize→JPEG.
    """
    image = image.convert("RGB")
    family = condition.family
    if family in {"resize", "resize_jpeg"}:
        width, height = image.size
        image = image.resize(
            (max(1, round(width * condition.factor)), max(1, round(height * condition.factor))),
            Image.Resampling.LANCZOS,
        )
    elif family == "crop":
        width, height = image.size
        side = math.sqrt(condition.factor)
        new_w, new_h = max(1, round(width * side)), max(1, round(height * side))
        left, top = (width - new_w) // 2, (height - new_h) // 2
        image = image.crop((left, top, left + new_w, top + new_h))
    elif family == "blur":
        image = image.filter(ImageFilter.GaussianBlur(radius=condition.radius))
    elif family == "sharpen":
        image = image.filter(
            ImageFilter.UnsharpMask(radius=condition.radius, percent=condition.percent, threshold=0)
        )
    elif family not in {"clean", "jpeg"}:
        raise ValueError(f"Unknown transformation family: {family}")
    if family in {"jpeg", "resize_jpeg"}:
        stream = BytesIO()
        image.save(
            stream,
            format="JPEG",
            quality=condition.quality,
            subsampling=2,
            optimize=False,
            progressive=False,
        )
        stream.seek(0)
        with Image.open(stream) as decoded:
            image = decoded.convert("RGB")
    return image
