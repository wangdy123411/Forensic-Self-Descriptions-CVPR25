# Comparison with the authors' implementation

The original implementation and AN agree on all **175 matched detection
decisions and source predictions**, with numerical differences well below the
declared tolerances. Read [the results](RESULTS.md) for interpretation and
[the full report](run1024/report.md) for case-level audits and plots.

This comparison supplements the earlier numerical validation, whose reference at
the default 1024 setting was the AN tiled extractor. Here, the authors' original
extraction function runs on the same PC and the same processed images as AN.

## Configuration and cohort

The source is official upstream main commit
`50f2eae06efdac2e5a33f407ca9a27a2295133ac`, with explicit v1.2.0 release weights.
The configuration retains float64, three scales, an 11 × 11 neighborhood,
`max_size=1024`, and released `resize_and_crop` preprocessing. The `fsd/` files
are unmodified. [Source verification](source_verification.md) records the exact
checkout and its differences from the earlier v1.2.0 source tag.

`manifest.csv` fixes 25 originals, five each from COCO, DALL-E 3, Firefly, SDXL,
and GLIDE. Selection uses a deterministic hash without detector scores.
`selection.json` preserves that procedure and the initial candidate grid. The
executed comparison uses seven conditions: clean, JPEG Q50, half-size resizing,
center crop to 25% area, blur radius 2, sharpening at 200%, and half-size resizing
followed by JPEG Q50. This gives 175 pairs.

Task metrics use this identical cohort on both sides. Comparing these 25-image
metrics with the 300-original main-study metrics would confound implementation
with sample selection. The small cohort is an implementation check, not a precise
estimate of population performance.

## What the comparison checks

1. Five public `FSDDetector.attribute` probes and two `score` checks exercise the
   released API. `audit_api_probes.py` compares the saved outputs with AN,
   converting raw source likelihoods through the release's temperature-750
   softmax when comparing probabilities.
2. The matched run calls original `compute_fsd`, projection, GMM, and attribution
   components. It compares raw, projected, and attribution features against both
   fresh AN extraction and saved main-study features.
3. The analyzer recomputes detection and attribution metrics on matched cases.
   Attribution accuracy includes only unambiguously mapped sources; predictions
   for GLIDE can still be checked for implementation agreement.

Declared limits are relative feature L2 error `1e-4`, absolute z-score difference
`1e-3`, and maximum probability difference `1e-4`. Decision and source-label
mismatches are reported separately. The results establish agreement within these
tolerances on the tested cohort, not bitwise equality on every possible image.

## Run a new comparison

Use a new output directory so the completed evidence remains intact:

```powershell
an/.venv/Scripts/python.exe an/baseline_comparison/compare_upstream.py --output an/baseline_comparison/repeat1024 --historical-source-archive an/provenance/pre_cleanup_source.zip --hours 6
an/.venv/Scripts/python.exe an/baseline_comparison/analyze_comparison.py an/baseline_comparison/repeat1024 --require-complete
```

The historical archive preserves the original source bytes referenced by the
frozen main-study hashes. The explicit archive option checks those hashes and
verifies that maintained inference code differs only in formatting or docstrings
before comparing with saved features. It does not permit changes in executable
logic. Without this option, the original strict source-hash check remains active.
Do not resume the completed original directory with changed source files.

Run metadata, coverage, errors, and features are written under the chosen output
directory. The completed study's [API audit](api_probe_audit.json),
[comparison summary](run1024/comparison_summary.json), and
[historical provenance](final_provenance.json) remain separate records.

## Runtime and memory interpretation

The original function materializes about 9.93 GiB of patch tensors at the default
grid. Despite exceeding the RTX 2060's 6 GiB physical memory, it completed on this
Windows machine. Logical CUDA allocations and observed behavior are consistent
with shared-memory fallback or paging, but memory residency was not traced.

AN avoids those large allocations. Timings use alternating method order and
retain allocator caches, so they describe paired execution in this machine
state. Logical tensor allocation is distinct from resident VRAM or total system
memory. A battery interruption overlaps five cases; all are retained, with
separate [timing strata](run1024/runtime_power_strata.csv) for context. The
reported efficiency gain does not imply improved detection robustness.
