# Experiment protocol

The AN study asks how ordinary image processing affects the released FSD
detector, source predictions, residuals, and 960-dimensional descriptors. The
processing grid and scoring rules were fixed before inspecting benchmark scores.
The checkpoint remained unchanged throughout.

## Why these tests

JPEG changes quantized image detail; resizing changes spatial sampling; cropping
removes content without resampling. Blur suppresses high-frequency detail, while
sharpening enhances it. Resize followed by JPEG tests a common sequence of
operations. Three settings per family provide a compact sensitivity sweep;
their nominal severities are not equivalent across families.

Half-size resizing and a crop retaining 25% of the area produce the same output
dimensions in this cohort. They provide a useful comparison between resampling
and content removal, although their difference does not isolate a single causal
mechanism. All transformations occur in RGB space before the released
grayscale and residual-extraction pipeline. The same settings are applied to
every real and synthetic test original.

## Fixed configuration

| Component | Setting |
|---|---|
| Source revision | Official main `50f2eae06efdac2e5a33f407ca9a27a2295133ac` |
| Weights | Explicit v1.2.0 release, recorded in `an/weights_provenance.json` |
| Residual extractor | 8 channels, 15 × 15 convolution |
| FSD | 11 × 11 neighborhoods excluding centers, 3 scales, 960 features |
| Residual preprocessing | Released `resize_and_crop`, internal grid size 1024 |
| Computation | Float64, FFT accumulation with reflection-boundary correction |
| Detection rule | Released `z < -2`; optional separate clean-real calibration |

FFT accumulation preserves the released normal equations, observations, and
constraints, with a different floating-point summation order. Numerical
validation and the direct upstream comparison are documented in
[implementation notes](implementation_notes.md).

## Processing grid

| Family | Settings | Implementation |
|---|---|---|
| Clean | One baseline | Decoded original RGB, no added processing |
| JPEG | Q95, Q75, Q50 | Pillow encode/decode, 4:2:0 subsampling, baseline, no optimization |
| Resize | Side factors 0.75, 0.50, 0.25 | One Lanczos downsample, no resize back |
| Crop | Retained areas 75%, 50%, 25% | Center crop with side factor equal to the square root of retained area; no resampling |
| Blur | Standard deviations 0.5, 1.0, 2.0 pixels | Pillow GaussianBlur |
| Sharpen | Amounts 50%, 100%, 200% | Pillow UnsharpMask, radius 1 pixel, threshold 0 |
| Resize then JPEG | 0.75 then Q75; 0.50 then Q75; 0.50 then Q50 | Same operations, in that order |

There are 19 conditions including clean. Pillow's blur and sharpening semantics
are defined in [ImageFilter](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html).
The model subsequently resizes and crops residuals internally, sometimes
upsampling small inputs. The experiment measures this complete released pipeline.

## Cohort and partitioning

The completed test cohort contains 100 COCO photographs and 50 images each from
Synthbuster's DALL-E 3, Firefly, SDXL, and GLIDE collections. A further 100 COCO
photographs are reserved for clean-only calibration. This produces 5,800 cases:
300 test originals under 19 conditions and 100 calibration originals under clean.

Images were selected by seeded hashes of archive identifiers, independently of
detector outputs. Original bytes, native formats, dimensions, source identifiers,
and hashes are retained. Every derivative keeps its original's partition.
The exact acquisition and selection procedures are in
[dataset_access.md](dataset_access.md). The local split is disjoint, but overlap
with the checkpoint authors' unpublished training images cannot be excluded.

## Outcomes and uncertainty

- **Detection:** pooled and per-generator AUROC and average precision,
  generator-macro AUROC, TPR, FPR, balanced accuracy, and decision flips.
  Synthetic is the positive class and the anomaly score is `-z`. Score direction
  remains fixed even when AUROC falls below chance.
- **Operating thresholds:** the released `z < -2` rule and the 95th percentile
  of separate clean-real calibration scores. Both stay fixed across test
  conditions; transformed test images never select a threshold.
- **Descriptor drift:** paired relative L2 and cosine distance for raw and
  detection-projected features, relative to the same clean original.
- **Residual diagnostics:** channel energy, native and post-preprocessing RMS,
  and high-frequency energy in a native center patch. Previews share a display
  scale within an original. Geometric variants are not presented as aligned
  pixelwise differences.
- **Attribution:** accuracy only for unambiguously mapped checkpoint sources.
  GLIDE predictions remain descriptive. No unknown-rejection claim is made.
- **Uncertainty:** 1,000 source-stratified bootstrap draws of original IDs,
  reusing draws for clean and processed scores. Report pointwise 95% intervals
  for AUROC and paired change. Derivatives are repeated measures, not independent
  images. A supplementary archive-stem grouping checks sensitivity to possible
  shared content without claiming verified semantic independence.
- **Execution:** complete coverage, explicit errors, runtime, allocated GPU
  memory, environment versions, and code/weight hashes.

## Interpretation

AUROC separates ranking failure from threshold sensitivity. Calibration may
help when ranking survives, but cannot repair a ranking reversal. Residual and
descriptor drift describe sensitivity; neither establishes a causal explanation
by itself. The transforms simulate processing workflows rather than measured
outputs from a named sharing platform. Mitigations and broader-domain tests
remain follow-up experiments, as discussed in [the results](../RESULTS.md).
