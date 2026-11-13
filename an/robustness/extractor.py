"""Memory-bounded FSD extraction with the released implementation's objective.

All calculations use float64. ``tiled`` materializes a few rows of patches at a
time. ``fft`` forms the circular-padded Gram matrix from spatial correlations,
then exactly replaces circular-padding boundary contributions with the released
reflect-padding contributions. The latter changes summation order, not the
features, scale weighting, ridge, constraints, or preprocessing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


@dataclass
class ExtractionResult:
    """Descriptor on CPU, numerical diagnostics, and optional residual images."""

    fsd: torch.Tensor
    diagnostics: dict[str, Any]
    residual_preview: np.ndarray | None = None
    residual_patch: np.ndarray | None = None


def _image_tensor(image, device):
    """Convert to a (1, H, W) float64 tensor without intensity normalization."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            image = opened.convert("L")
    if isinstance(image, Image.Image):
        image_t = torch.from_numpy(np.array(image.convert("L"))).double().unsqueeze(0)
    elif isinstance(image, torch.Tensor):
        image_t = image.double()
        if image_t.ndim == 2:
            image_t = image_t.unsqueeze(0)
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")
    if image_t.ndim != 3 or image_t.shape[0] != 1:
        raise ValueError("FSD requires a grayscale tensor with shape (1, H, W).")
    return image_t.to(device)


def _prepare_residuals(image, fre, max_size, resize_mode, capture_native=False):
    """Apply the released residual extraction, border removal, and resizing.

    Native diagnostics precede residual resizing; the returned residual tensor
    uses the released spatial preprocessing. These are distinct measurements.
    """
    if resize_mode not in {"resize", "crop", "resize_and_crop"}:
        raise ValueError(f"Unsupported resize_mode: {resize_mode}")
    image_t = _image_tensor(image, fre.device)
    border = fre.conv.kernel_size // 2
    if min(image_t.shape[-2:]) <= 2 * border:
        raise ValueError("Image is smaller than the FRE's valid spatial support.")
    residuals = fre(image_t)
    if border:
        residuals = residuals[:, border:-border, border:-border]
    raw_rms = torch.mean(residuals.square(), dim=(-2, -1)).sqrt().cpu().tolist()
    raw_shape = list(residuals.shape[-2:])
    native_diagnostics = {}
    native_patch = None
    if capture_native:
        ph, pw = min(128, raw_shape[0]), min(128, raw_shape[1])
        sy, sx = (raw_shape[0] - ph) // 2, (raw_shape[1] - pw) // 2
        native = residuals[:, sy : sy + ph, sx : sx + pw]
        native_patch = native.cpu().numpy().copy()
        # Bound diagnostic FFT cost independently of the native image size.
        # Demeaning and a separable Hann window reduce DC/boundary leakage.
        window = torch.hann_window(
            ph, periodic=False, device=residuals.device, dtype=torch.float64
        )[:, None]
        window = (
            window
            * torch.hann_window(pw, periodic=False, device=residuals.device, dtype=torch.float64)[
                None, :
            ]
        )
        spectrum = torch.fft.fft2((native - native.mean((-2, -1), keepdim=True)) * window)
        energy = spectrum.abs().square()
        fy = torch.fft.fftfreq(ph, device=residuals.device, dtype=torch.float64)
        fx = torch.fft.fftfreq(pw, device=residuals.device, dtype=torch.float64)
        high = (fy[:, None].square() + fx[None, :].square()).sqrt() >= 0.25
        total = energy.sum((-2, -1))
        high_energy = energy[:, high].sum(1)
        native_diagnostics = {
            "native_residual_patch_shape": [ph, pw],
            "native_residual_patch_hf_cutoff_cycles_per_pixel": 0.25,
            "native_residual_patch_hf_fraction": float(
                high_energy.sum() / total.sum().clamp_min(1e-30)
            ),
            "native_residual_patch_hf_fraction_channels": (high_energy / total.clamp_min(1e-30))
            .cpu()
            .tolist(),
        }
    if "resize" in resize_mode:
        h, w = residuals.shape[-2:]
        factor = max_size / min(h, w)
        residuals = F.interpolate(
            residuals[None],
            size=(round(h * factor), round(w * factor)),
            mode="bilinear",
            antialias=False,
            align_corners=False,
        )[0]
    if "crop" in resize_mode:
        h, w = residuals.shape[-2:]
        crop_h, crop_w = min(h, max_size), min(w, max_size)
        start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
        residuals = residuals[:, start_h : start_h + crop_h, start_w : start_w + crop_w]
    diagnostics = {
        "input_height": int(image_t.shape[-2]),
        "input_width": int(image_t.shape[-1]),
        "raw_residual_shape": raw_shape,
        "raw_residual_rms_channels": raw_rms,
        "raw_residual_rms": float(np.sqrt(np.mean(np.square(raw_rms)))),
    }
    diagnostics.update(native_diagnostics)
    if capture_native:
        return residuals, diagnostics, native_patch
    return residuals, diagnostics


def _scaled_residuals(residuals, kernel_size, num_scales):
    """Yield the released bilinear pyramid without changing scale weights."""
    for level in range(num_scales):
        scaled = F.interpolate(
            residuals[None],
            scale_factor=1 / 2**level,
            mode="bilinear",
            antialias=False,
            align_corners=False,
        )[0]
        if min(scaled.shape[-2:]) <= kernel_size // 2:
            raise ValueError(f"Scale {level} is too small for reflection padding.")
        yield scaled


def _patch_matrix(patches, mask, center):
    # patches are (N, K, B, B), ordered channel-major within each sample.
    x = patches[:, :, mask].reshape(patches.shape[0], -1)
    y = patches[:, :, center, center].sum(dim=1)
    return x, y


def _accumulate_tiled(residuals, kernel_size, num_scales, tile_rows):
    """Accumulate X.T @ X and X.T @ y over bounded rows of reflected patches."""
    channels = residuals.shape[0]
    center = kernel_size // 2
    mask = torch.ones(kernel_size, kernel_size, dtype=torch.bool, device=residuals.device)
    mask[center, center] = False
    dim = channels * (kernel_size * kernel_size - 1)
    gram = residuals.new_zeros((dim, dim))
    rhs = residuals.new_zeros(dim)
    samples = 0
    for scaled in _scaled_residuals(residuals, kernel_size, num_scales):
        h, w = scaled.shape[-2:]
        padded = F.pad(scaled[None], (center,) * 4, mode="reflect")[0]
        windows = padded.unfold(1, kernel_size, 1).unfold(2, kernel_size, 1)
        for start in range(0, h, tile_rows):
            slab = windows[:, start : start + tile_rows]
            patches = slab.reshape(channels, -1, kernel_size, kernel_size).permute(1, 0, 2, 3)
            x, y = _patch_matrix(patches, mask, center)
            gram.addmm_(x.T, x)
            rhs.addmv_(x.T, y)
        samples += h * w
    return gram, rhs, samples


def _accumulate_fft(residuals, kernel_size, num_scales, boundary_chunk):
    """Accumulate the same normal equations using FFT correlations.

    Correlations initially describe circular boundaries. The explicit boundary
    pass subtracts those patches and adds reflected patches, matching upstream
    padding while avoiding an all-image patch matrix.
    """
    channels = residuals.shape[0]
    device = residuals.device
    center = kernel_size // 2
    mask = torch.ones(kernel_size, kernel_size, dtype=torch.bool, device=device)
    mask[center, center] = False
    offsets = mask.nonzero() - center
    n_features = len(offsets)
    dim = channels * n_features
    gram = residuals.new_zeros((dim, dim))
    rhs = residuals.new_zeros(dim)
    # Correlation lag is column offset minus row offset.
    delta = offsets[None, :, :] - offsets[:, None, :]
    samples = 0
    for scaled in _scaled_residuals(residuals, kernel_size, num_scales):
        h, w = scaled.shape[-2:]
        transforms = torch.fft.rfft2(scaled)
        lag_y, lag_x = delta[..., 0].remainder(h), delta[..., 1].remainder(w)
        neg_y, neg_x = (-offsets[:, 0]).remainder(h), (-offsets[:, 1]).remainder(w)
        pos_y, pos_x = offsets[:, 0].remainder(h), offsets[:, 1].remainder(w)
        for k in range(channels):
            ks = slice(k * n_features, (k + 1) * n_features)
            for j in range(k, channels):
                js = slice(j * n_features, (j + 1) * n_features)
                correlation = torch.fft.irfft2(transforms[k].conj() * transforms[j], s=(h, w))
                block = correlation[lag_y, lag_x]
                gram[ks, js] += block
                rhs[ks] += correlation[neg_y, neg_x]
                if j != k:
                    gram[js, ks] += block.T
                    rhs[js] += correlation[pos_y, pos_x]
        del transforms

        # Only patch centers within radius B//2 of an edge differ between
        # circular and reflect padding. Corners are counted exactly once.
        boundary = torch.zeros((h, w), dtype=torch.bool, device=device)
        if center:
            boundary[:center, :] = True
            boundary[-center:, :] = True
            boundary[:, :center] = True
            boundary[:, -center:] = True
        yy, xx = boundary.nonzero(as_tuple=True)
        reflected = F.pad(scaled[None], (center,) * 4, mode="reflect")[0]
        circular = F.pad(scaled[None], (center,) * 4, mode="circular")[0]
        reflected_windows = reflected.unfold(1, kernel_size, 1).unfold(2, kernel_size, 1)
        circular_windows = circular.unfold(1, kernel_size, 1).unfold(2, kernel_size, 1)
        for start in range(0, len(yy), boundary_chunk):
            ys, xs = yy[start : start + boundary_chunk], xx[start : start + boundary_chunk]
            rp = reflected_windows[:, ys, xs].permute(1, 0, 2, 3)
            cp = circular_windows[:, ys, xs].permute(1, 0, 2, 3)
            xr, yr = _patch_matrix(rp, mask, center)
            xc, yc = _patch_matrix(cp, mask, center)
            gram.addmm_(xc.T, xc, alpha=-1)
            gram.addmm_(xr.T, xr)
            rhs.addmv_(xc.T, yc, alpha=-1)
            rhs.addmv_(xr.T, yr)
        samples += h * w
    return gram, rhs, samples


def _solve(gram, rhs, channels, lambda_reg):
    """Solve the ridge-regularized KKT system with one unit-sum constraint per channel.

    The diagonal update modifies ``gram`` in place. Both callers pass an
    accumulator that is no longer needed after this solve.
    """
    dim = len(rhs)
    n_features = dim // channels
    gram.diagonal().add_(lambda_reg)
    constraint = gram.new_zeros((channels, dim))
    for channel in range(channels):
        constraint[channel, channel * n_features : (channel + 1) * n_features] = 1
    system = torch.cat(
        [
            torch.cat([gram, constraint.T], dim=1),
            torch.cat([constraint, gram.new_zeros((channels, channels))], dim=1),
        ]
    )
    target = torch.cat([rhs, rhs.new_ones(channels)])
    solution = torch.linalg.solve(system, target)
    fsd = solution[:dim]
    diagnostics = {
        "constraint_max_abs_error": float((constraint @ fsd - 1).abs().max().item()),
        "kkt_relative_residual": float(
            ((system @ solution - target).norm() / target.norm().clamp_min(1e-30)).item()
        ),
    }
    return fsd, diagnostics


@torch.no_grad()
def extract_fsd(
    image,
    fre,
    kernel_size=11,
    num_scales=3,
    max_size=1024,
    resize_mode="resize_and_crop",
    method="tiled",
    tile_rows=8,
    boundary_chunk=2048,
    lambda_reg=1e-5,
    return_preview=False,
):
    """Compute the released FSD descriptor without all-image patch storage.

    ``fft`` is mathematically equivalent but must be validated against tiled
    and upstream extraction for the weights/hardware in use before a study.
    Diagnostics describe residual energy before and after the released
    residual resize/crop. Previews are 8x64x64 by default (all FRE channels).
    """
    if kernel_size < 3 or kernel_size % 2 != 1 or num_scales < 1:
        raise ValueError("kernel_size must be odd and >=3; num_scales must be >=1.")
    if tile_rows < 1 or boundary_chunk < 1 or max_size < 1 or lambda_reg < 0:
        raise ValueError("Tile/chunk/max_size must be positive and ridge nonnegative.")
    if method not in {"tiled", "fft"}:
        raise ValueError("method must be 'tiled' or 'fft'.")
    if fre.device.type == "cuda":
        torch.cuda.synchronize(fre.device)
    started = time.perf_counter()
    residuals, diagnostics, native_patch = _prepare_residuals(
        image, fre, max_size, resize_mode, capture_native=True
    )
    diagnostics["residual_shape"] = list(residuals.shape[-2:])
    diagnostics["residual_rms_channels"] = (
        residuals.square().mean(dim=(-2, -1)).sqrt().cpu().tolist()
    )
    diagnostics["residual_rms"] = float(residuals.square().mean().sqrt().item())
    diagnostics["residual_abs_mean"] = float(residuals.abs().mean().item())
    preview = None
    if return_preview:
        preview = F.interpolate(residuals[None], size=(64, 64), mode="area")[0].cpu().numpy()
    if method == "tiled":
        gram, rhs, samples = _accumulate_tiled(residuals, kernel_size, num_scales, tile_rows)
    else:
        gram, rhs, samples = _accumulate_fft(residuals, kernel_size, num_scales, boundary_chunk)
    fsd, solve_diagnostics = _solve(gram, rhs, residuals.shape[0], lambda_reg)
    diagnostics.update(solve_diagnostics)
    diagnostics.update(
        {
            "method": method,
            "dtype": "float64",
            "samples": samples,
            "feature_dim": int(fsd.numel()),
            "lambda_reg": lambda_reg,
            "num_scales": num_scales,
            "max_size": max_size,
        }
    )
    fsd = fsd.cpu()
    if not torch.isfinite(fsd).all():
        raise FloatingPointError("Nonfinite FSD descriptor.")
    if fre.device.type == "cuda":
        torch.cuda.synchronize(fre.device)
    diagnostics["seconds"] = time.perf_counter() - started
    return ExtractionResult(fsd, diagnostics, preview, native_patch if return_preview else None)


def compute_fsd_tiled(
    image,
    fre,
    kernel_size=11,
    num_scales=3,
    max_size=1024,
    resize_mode="resize_and_crop",
    tile_rows=8,
):
    """Drop-in equivalent of fsd.fsd_computation.compute_fsd."""
    return extract_fsd(
        image,
        fre,
        kernel_size,
        num_scales,
        max_size,
        resize_mode,
        method="tiled",
        tile_rows=tile_rows,
    ).fsd
