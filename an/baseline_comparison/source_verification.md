# Pinned upstream source verification

The direct baseline is **official upstream main commit
`50f2eae06efdac2e5a33f407ca9a27a2295133ac`, using explicit v1.2.0 release
weights**. It is not the exact source tree referenced by the lightweight
`v1.2.0` tag.

The root agent's successful read-only `git ls-remote` observation returned:

| Official reference | Commit |
| --- | --- |
| HEAD | `50f2eae06efdac2e5a33f407ca9a27a2295133ac` |
| refs/heads/main | `50f2eae06efdac2e5a33f407ca9a27a2295133ac` |
| refs/tags/v1.2.0 | `5b317a00251988b5ec5a47317f4d82e5bdfd009d` |

The tag is lightweight; no peeled tag reference was returned. Its commit is an
ancestor of main, as independently checked with the local Git object database.
The local fork checkout has the same HEAD as the observed official main, and
its entire `fsd/` working tree is clean.

The offline [verification script](verify_source.py) exported all 11 package
files from both commits. For every main file, its Git blob SHA-1 agrees with
both an independently calculated Git-object hash and the working file hashed
through Git's configured filters. The [machine-readable evidence](source_verification.json)
contains every blob ID, SHA-256, and snapshot path; [the exact package diff](tag_to_main_fsd.patch)
shows the tag-to-main changes. The canonical snapshots preserve exact Git
bytes. The Windows checkout has `core.autocrlf=true`, so raw working-file
bytes can differ by newline encoding; the JSON reports that comparison
separately. This is stronger and more precise than asserting raw Windows
bytes equal LF-encoded upstream blobs.

The official remote-reference observation was performed by the root agent
after this subagent's network escalation stalled. That distinction is recorded
in the JSON; the offline script does not pretend to have fetched remote refs
itself. Primary source locations are the [pinned official commit](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/commit/50f2eae06efdac2e5a33f407ca9a27a2295133ac),
[official tag tree](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/tree/v1.2.0/fsd),
and [v1.2.0 release](https://github.com/ductai199x/Forensic-Self-Descriptions-CVPR25/releases/tag/v1.2.0).

## Material tag-versus-main differences

Only three of the 11 `fsd/` files differ:

| File | Tag commit 5b317a0 | Main commit 50f2eae |
| --- | --- | --- |
| `__init__.py` | Package version string 1.1.0 | Version string 1.2.0 |
| `weights.py` | Automatic cache/download pins v1.1.0 | Pins v1.2.0 |
| `attribution.py` | Loads source GMM plus per-source likelihood mean/std; selects the largest normalized likelihood and softmaxes those normalized values | Loads source GMMs directly; selects largest raw log likelihood; confidence uses temperature-750 softmax over raw likelihood differences |

The attribution change can alter **predicted classes**, not just confidence.
Blindly checking out the tag and accepting its automatic download defaults
would also select a different weight release. Do not conflate those changes
with the AN extraction optimization.

The canonical blobs for `fre.py`, `fsd_computation.py`, `detector.py`,
`projection.py`, and `gmm.py` are identical between tag and main. The
remaining unchanged files are `cli.py`, `cli_ray.py`, and `serve.py`.
The running baseline uses main's original attribution rule and the exact
same explicit weight directory as AN.

## Defaults and a fair API comparison

- `FSDDetector.load` defaults to `device="auto"`, no explicit weight path,
  the configuration's threshold, and `attribution=False`. The comparison
  explicitly selects CUDA, `an/weights/v1.2.0`, and attribution support.
- The loaded configuration uses an eight-channel 15-by-15 FRE; 11-by-11 FSD
  neighborhoods excluding centers; three scales; a 1024 residual grid;
  `resize_and_crop`; and a five-component tied-covariance detection GMM.
  These agree with the main AN experiment.
- Original `compute_fsd` converts PIL input to grayscale, retains the
  0–255 scale, and computes in float64. Residual normalization resizes the
  shorter side to 1024, including upsampling, then center-crops. It does not
  merely cap large RGB inputs at 1024.
- The original extraction materializes all multiscale overlapping patches
  before accumulating normal equations in chunks of 16,384. Its ridge is
  `1e-5`, with one sum-to-one constraint per residual channel.
- Detection uses `z=(raw_log_likelihood-train_mean)/train_std`, where the
  loaded mean is `-42.25325127289017` and standard deviation is
  `706.0556010649537`. The default decision is strictly `z < -2`.
  AN's positive/fake anomaly is `-z`; this sign conversion is not a model change.
- `attribute` applies the detection transforms before attribution transforms,
  computes detection and 14-source output in one extraction pass, and returns
  **probabilities** in `scores`. AN saves raw source log likelihoods; compare
  them to API probabilities only after the original temperature-750 softmax.
  There is no unknown-rejection branch or detection gate suppressing labels.
- Calling `score` and then `attribute` on the same image repeats extraction.
  `score_batch` is a Python loop. The timing comparison must use equivalent
  workloads and distinguish original API extraction from AN diagnostics/cache
  overhead.
- Both baseline and AN must receive identical processed RGB pixels and use
  the same frozen originals, conditions, source mapping, and thresholds.
  The 25-original comparison cohort cannot be compared directly against the
  300-original main result to infer implementation effects.

## Training and numerical-validation boundaries

This package-source check establishes source identity with the observed
official main commit. It does not establish equivalence to the CVPR paper's
training procedure or unpublished data splits. The inspected release provides
inference and GMM routines, not a complete end-to-end FRE/transform training
pipeline with image-level training manifests. Public evaluation overlap with
checkpoint training cannot be ruled out from source identity alone.

Earlier 1024-grid validation compared FFT extraction with the **AN tiled
reference**, while original-function comparisons in those initial tests used
tractable smaller inputs. The new default-1024 probes and matched comparison
are a separate, literal upstream execution; they must not be retroactively
described as what the earlier audit measured.

Literal default-1024 API probes have now succeeded on this Windows RTX 2060
system despite allocations exceeding its physical VRAM. The first COCO probe
took 93.3 seconds and reported approximately 10.7 GiB of logical CUDA
allocation. Shared-memory fallback/paging is a plausible explanation, not a
directly measured residency trace. AN avoids these oversized allocations;
the evidence no longer supports saying the optimization was strictly required
to run the original here. See [implementation notes](../docs/implementation_notes.md)
and [the comparison protocol](README.md).
