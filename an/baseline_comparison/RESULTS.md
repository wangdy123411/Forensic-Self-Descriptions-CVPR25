# Original implementation versus AN

**The original implementation and AN agree on all 175 matched detection decisions and source predictions.** Numerical differences are well below the declared tolerances. Processing failures persist in the authors' implementation on this subset, while AN reduces runtime and tensor allocation. The comparison completed on 8 November 2026 using the Windows RTX 2060 and `an/.venv`.

## Configuration and cohort

The source is [official upstream main commit `50f2eae06efdac2e5a33f407ca9a27a2295133ac`](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/tree/50f2eae06efdac2e5a33f407ca9a27a2295133ac), with the same explicit **v1.2.0 release weights** as the main AN study. All 11 local `fsd` files match this commit after Git newline filtering. The earlier `v1.2.0` source tag is a different commit with different attribution/default-weight behavior; this comparison does not claim to execute that tag. See [source verification](source_verification.md).

The 25 frozen test originals comprise five COCO photographs and five images from each of DALL-E 3, Firefly, SDXL, and GLIDE. Selection used a deterministic hash without scores. Seven conditions produce **175 paired cases**: clean, JPEG Q50, half-size resizing, center crop to 25% area, blur radius 2, sharpening 200%, and half-size resizing followed by JPEG Q50.

The original extraction function ran at its default configuration: float64, 11×11 neighborhood, three scales, `max_size=1024`, and `resize_and_crop`. Each processed image was shared with a fresh AN FFT extraction. Both were also compared with the saved AN main-study features and predictions.

Five additional `FSDDetector.attribute()` probes and two `score()` checks passed. The 175-case run calls original extraction and scoring components, rather than the full public `attribute()` wrapper. The original 5,800 main-study prediction records remain unchanged.

## Numerical and task agreement

| Check | Result |
|---|---:|
| Completed cases / errors | 175 / 0 |
| Released detection decisions matching | 175/175 |
| Frozen calibrated decisions matching | 175/175 |
| Source predictions matching | 175/175 |
| Maximum raw 960D relative L2 difference | 3.40×10⁻⁷ |
| Maximum absolute z-score difference | 3.54×10⁻⁷ |
| Maximum source-probability difference | 1.82×10⁻⁷ |
| Fresh AN versus saved AN feature/score differences | 0 in recorded values |

Declared limits were 10⁻⁴ relative L2 for raw, detection-projected, and attribution-projected descriptors; 10⁻³ absolute z difference; and 10⁻⁴ maximum probability difference. The upstream/AN result is agreement within these tolerances, not bitwise identity or a proof for every possible input.

Every task metric is recomputed on the **same 25 originals and conditions** for both implementations:

| Condition | Original macro AUROC | AN macro AUROC |
|---|---:|---:|
| Clean | 0.99 | 0.99 |
| JPEG Q50 | 1.00 | 1.00 |
| Resize 0.5× | 0.66 | 0.66 |
| Crop to 25% area | 0.67 | 0.67 |
| Blur radius 2 | 1.00 | 1.00 |
| Sharpen 200% | 0.99 | 0.99 |
| Resize 0.5× → JPEG Q50 | 0.58 | 0.58 |

Macro AUROC averages the four real-versus-generator AUCs. Released-threshold clean TPR/FPR is 95%/0%; resize→JPEG yields 45%/20%. Both implementations give exactly the same values. With only five real originals, a single false positive changes FPR by 20 percentage points; these figures validate agreement rather than provide a precise population estimate.

Among the 15 supported synthetic-source originals, attribution accuracy is 40% clean, 6.7% at JPEG Q50, 0% after half-size resizing, 0% after cropping, 0% after blur, 46.7% after sharpening, and 0% after resize→JPEG, identically for both implementations. This accuracy excludes COCO and unmapped GLIDE; the full classifier still predicts among all 14 released labels. [Detailed results](run1024/report.md) include TPR/FPR, both attribution cohorts, and per-source metrics.

## Runtime and memory

| Measurement, all 175 pairs | Original | Fresh AN FFT |
|---|---:|---:|
| Median measured seconds per case | 44.765 | 2.936 |
| Maximum PyTorch allocated memory | 10.76 GiB | 0.811 GiB |

The **median per-case speed ratio is 15.07×**. Timers include synchronized extraction, original scoring, CPU output copies, and the same extra source-likelihood diagnostic pass. AN also includes its production residual diagnostics. They exclude image processing, file writing, and the extra public-API probes. Method order alternated; allocator caches were retained, so these are paired timings in this machine state, not isolated cold-start benchmarks.

The original implementation completed despite logical allocations exceeding the GPU's 6 GiB physical VRAM. Observations are consistent with Windows shared-memory fallback or paging, but residency was not traced. The table reports logical tensor allocation, not resident VRAM. AN avoids the oversized patch allocations. These results establish that the original default configuration is feasible on this machine; the earlier numerical audit used AN tiled extraction as its reference.

A battery interruption overlaps five cases. All remain in the overall timings and every accuracy/numerical comparison. For the 170 cases outside the recorded incident window, medians are 44.743 seconds upstream and 2.932 seconds AN, with a median paired speed ratio of 15.05×. Outside-window status is not proof of uninterrupted AC power. See [timing strata](run1024/runtime_power_strata.csv) and [incident evidence](power_incident.json).

Paired scoring and analysis ran from approximately **16:54 to 19:21 Singapore time**. All **38 automated tests passed** at completion. The [audit JSON](run1024/comparison_summary.json) and [historical provenance inventory](final_provenance.json) preserve the checks and artifacts.

## Practical implications

AN FFT makes the larger robustness sweep practical on this GPU while preserving the tested numerical behavior. Resizing and cropping losses, threshold sensitivity, and weak attribution remain present in the original implementation. The [300-original robustness study](../RESULTS.md) provides the broader performance evidence; this smaller matched cohort validates the implementation and measures efficiency.

The 175 derivatives come from 25 originals, and overlap with checkpoint-training images remains unknown. The comparison uses the specified main commit and released weights; it does not reconstruct the paper's training or full benchmark. Controlled transformations also differ from measured platform outputs. The next priorities are a larger held-out geometric-processing test, an additional real-image domain, and separate validation of attribution and processed-image calibration.
