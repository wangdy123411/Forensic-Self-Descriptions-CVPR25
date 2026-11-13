from __future__ import annotations

import torch

from fsd.detector import DetectionResult
from fsd.projection import apply_projections


def raw_dimensions(detector) -> int:
    return int(detector.config["fre"]["out_channels"]) * (
        int(detector.config["fsd"]["kernel_size"]) ** 2 - 1
    )


def _validate_raw(detector, raw: torch.Tensor) -> None:
    if not isinstance(raw, torch.Tensor):
        raise TypeError("Raw FSD must be a torch.Tensor")
    if raw.dtype != torch.float64 or raw.ndim != 1 or raw.numel() != raw_dimensions(detector):
        raise ValueError("Raw FSD must be an unprojected, one-dimensional float64 vector of the configured size")
    if not bool(torch.isfinite(raw).all()):
        raise ValueError("Raw FSD contains nonfinite values")


def score_from_raw(detector, raw: torch.Tensor) -> DetectionResult:

    _validate_raw(detector, raw)
    device = detector.fre.device
    fsd_vec = raw.to(device).unsqueeze(0)
    with torch.no_grad():
        fsd_vec = apply_projections(fsd_vec, detector.projections)
    raw_score = detector.gmm.score_samples(fsd_vec).item()
    z_score = (raw_score - detector.train_mean) / detector.train_std
    return DetectionResult(
        z_score=z_score,
        raw_score=raw_score,
        is_fake=z_score < detector.threshold,
        threshold=detector.threshold,
    )


def score_and_cache(detector, path, item: dict, cache) -> tuple[DetectionResult, bool]:

    cached = cache.get(item)
    cache_hit = cached is not None
    if cache_hit:
        raw = torch.from_numpy(cached)
    else:
        raw = detector.compute_fsd(path)
        _validate_raw(detector, raw)
        if raw.device.type != "cpu":
            raise ValueError("Released raw FSD extraction must return a CPU tensor")
        cache.put(item, raw.detach().numpy(), phase="online")
    return score_from_raw(detector, raw), cache_hit
