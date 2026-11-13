# AN: image-processing robustness of FSD

This study examines how JPEG compression, resizing, cropping, blur, sharpening,
and resize-then-JPEG processing affect Forensic Self-Descriptions. It evaluates
the released checkpoint on matched image variants, measures changes in residuals
and 960-dimensional descriptors, and checks the optimized extractor against the
authors' implementation.

The completed study contains **5,800 cases from 400 original images**, including
100 photographs reserved for calibration. Resizing and cropping caused the
largest detection losses. A separate comparison found matching detection and
source predictions for all **175 pairs** evaluated with the original and AN
implementations, with a median paired speed ratio of **15.07×** on this RTX 2060.

## Read the reports

| Document | Purpose |
|---|---|
| [Report collection](reports/index.html) | Local browsing of the PDF, source, figures, and supporting evidence |
| [Results](RESULTS.md) | Main findings, limitations, and next experiments |
| [CVPR subsection](paper_cvpr2026/README.md) | Final PDF, LaTeX source, table, and figure |
| [Full robustness report](results/main/report.md) | All 19 conditions, uncertainty intervals, and residual galleries |
| [Original-code comparison](baseline_comparison/RESULTS.md) | Matched accuracy, numerical agreement, runtime, and memory |
| [Experiment protocol](docs/experiment_plan.md) | Processing settings, sample selection, and metrics |
| [Implementation notes](docs/implementation_notes.md) | FFT derivation, validation, and measurement definitions |
| [Dataset protocol](docs/dataset_access.md) | Acquisition, frozen subsets, and provenance |
| [Research audit](docs/research.md) | Paper, upstream release, related work, and original datasets |

## Code and data

| Path | Contents |
|---|---|
| `robustness/` | Processing, feature extraction, scoring, data acquisition, and analysis |
| `tests/` | Offline regression tests and numerical validation scripts |
| `data/` | Frozen manifests, acquisition records, and original images |
| `results/` | Completed phase runs, merged study, and numerical audits |
| `baseline_comparison/` | Original-code comparison scripts and results |
| `paper_cvpr2026/` | Paper subsection and publication assets |
| `docs/` | Methods and research documentation |
| `reports/`, `provenance/` | Portable report bundle, cleanup checks, and historical source archives |
| `weights/`, `.venv/` | Released checkpoint and isolated Python environment |

The upstream `fsd/` package is imported without modification. The study uses
official main commit `50f2eae06efdac2e5a33f407ca9a27a2295133ac` with explicit
v1.2.0 release weights. That commit differs from the earlier v1.2.0 source tag;
[source verification](baseline_comparison/source_verification.md) records the
differences. This is a released-checkpoint evaluation, not a reproduction of the
paper's training pipeline or complete 96-comparison benchmark.

## Reproduction environment

Run commands from the repository root. The existing environment uses Python 3.12:

```powershell
an/.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
```

To create the environment on another machine with Python 3.12:

```powershell
python -m venv an/.venv
an/.venv/Scripts/python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128 --cache-dir an/.cache/pip
an/.venv/Scripts/python.exe -m pip install -r an/requirements-lock.txt --cache-dir an/.cache/pip
an/.venv/Scripts/python.exe an/download_weights.py
```

[requirements-lock.txt](requirements-lock.txt) records the study environment;
[requirements.txt](requirements.txt) gives broader dependency constraints for
porting. CUDA runs fail explicitly if no GPU is available.

## Reproduce the study

The test set contains 100 COCO photographs and 50 images from each of DALL-E 3,
Firefly, SDXL, and GLIDE. A further 100 disjoint COCO photographs provide
clean-only calibration. Every test original receives 19 conditions, giving
300 × 19 + 100 = 5,800 cases. Follow the [dataset instructions](docs/dataset_access.md)
to acquire and freeze the two phase manifests, then use new output directories:

```powershell
an/.venv/Scripts/python.exe an/run_experiment.py --manifest an/data/manifest_real.csv --output an/results/real_repeat --device cuda --method fft --calibration-count 100 --hours 1.2
an/.venv/Scripts/python.exe an/run_experiment.py --manifest an/data/manifest_fake.csv --output an/results/fake_repeat --device cuda --method fft --calibration-count 100 --hours 2
an/.venv/Scripts/python.exe an/merge_runs.py an/results/real_repeat an/results/fake_repeat --output an/results/main_repeat
an/.venv/Scripts/python.exe an/analyze_results.py an/results/main_repeat --bootstrap 1000
```

Scoring resumes completed records when the configuration is unchanged. Changes
to weights, image selection, processing, or extraction code require a new output
directory. The time limit is checked between originals so that each original's
19-condition group can finish. A file named `STOP` in a scoring directory
requests a graceful stop after the current original. Merged directories contain
analysis outputs and cannot resume scoring. Use `--limit-per-source N` for a
smaller pilot while retaining the default model configuration.

## Verify and interpret

```powershell
an/.venv/Scripts/python.exe -m unittest discover -s an/tests
an/.venv/Scripts/python.exe an/tests/validate_extractor.py --weights an/weights/v1.2.0 --device cuda --max-size 1024 --skip-upstream --output an/results/extractor_validation_rerun.json
```

The second command checks numerical agreement with the AN tiled reference.
The separate [original-code protocol](baseline_comparison/README.md) runs the
authors' extraction function at its default 1024 setting. Numerical fixtures
measure implementation agreement; accuracy comes from the frozen image manifests.

Each run preserves its configuration, source hashes, predictions, cached feature
vectors, timing, and errors. Large images, model weights, environments, and
per-image arrays are ignored by Git but remain necessary for exact local
reproduction. Historical provenance snapshots describe the files at execution
time and should not be overwritten after documentation or code maintenance.
Pre-cleanup source and documentation are retained in
`provenance/pre_cleanup_source.zip`.

The [maintenance record](docs/maintenance.md) distinguishes the preserved study
from later source and documentation cleanup. To refresh the portable report
bundle without rerunning the experiments:

```powershell
an/.venv/Scripts/python.exe an/export_reports.py
```

The local calibration and test sets are disjoint, but the released checkpoint's
training image IDs are unavailable. Processing tests simulate editing workflows;
they do not measure a particular social-media service. Source attribution is a
closed-set prediction without unknown rejection. COCO and the selected Firefly
files already contain JPEG encoding, so “clean” means no processing added here.

Upstream code and weights use [CC BY-NC-SA 4.0](reports/UPSTREAM_LICENSE.txt). Cite FSD and the
datasets used, and retain their access and licensing terms.
