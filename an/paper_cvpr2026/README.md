# CVPR 2026 experiments subsection

The [final PDF](output/pdf/robustness_subsection.pdf) presents the robustness study and the comparison with the authors' implementation. The subsection, table, and figure occupy roughly three-quarters of a page. Three references complete the standalone page.

The document uses the unmodified official CVPR 2026 template, pinned to commit `12909ae437f6dbc7435069cfdb4ca44c18e6a02f`, with the standard 10-point font and two-column layout. The wrapper uses ragged column bottoms so the short excerpt is not stretched to fill the page.

## Files

| File | Purpose |
| --- | --- |
| [subsection.tex](subsection.tex) | Text and floats to insert into the Experiments section |
| [main.tex](main.tex) | Standalone document wrapper |
| [processing_table.tex](processing_table.tex) | Seven representative settings from the full sweep |
| [figures/generator_sensitivity.pdf](figures/generator_sensitivity.pdf) | Vector figure for publication |
| [references.bib](references.bib) | The three cited works |
| [output/robustness_latex_source.zip](output/robustness_latex_source.zip) | Source bundle with the required table, figure, and LaTeX dependencies |
| [make_assets.py](make_assets.py) | Regenerate the table and figure from saved experiment results |
| [build.py](build.py) | Compile the PDF and render a preview when Poppler is available |

The PNG figure is provided for slides and quick viewing. `texmf/` contains unchanged LaTeX packages needed by this PC; a complete TeX Live installation or Overleaf normally supplies them already.

## Scope and interpretation

The processing study uses 300 test originals and 19 conditions. The separate implementation comparison uses 25 originals and seven conditions, giving 175 matched cases. These cohorts answer different questions and are not compared as though they were the same benchmark.

The faster implementation changes descriptor accumulation while retaining the released model, preprocessing, float64 precision, and three scales. It improves computational efficiency; the study does not demonstrate a robustness mitigation. The processing tests simulate common edits rather than measure transformations collected from social-media platforms.

The seven table rows illustrate the full sweep rather than identify optimal settings. Half-size resizing and a 25%-area center crop produce equal dimensions for every test original, but differ in both content and sampling. Their comparison does not isolate interpolation as a cause. Exact settings and the sampling protocol are recorded in the [experiment plan](../docs/experiment_plan.md).

The table uses the released decision threshold. Plot intervals are pointwise, based on 1,000 source-stratified bootstrap draws of original IDs with matched multiplicities across conditions. Descriptor drift is the median relative L2 change in the raw 960-dimensional vector. Residual attenuation is the median of each image's processed-to-clean RMS ratio after internal preprocessing. GPU memory figures describe logical PyTorch allocations, not physical VRAM residency. See the [study results](../RESULTS.md) and [implementation comparison](../baseline_comparison/RESULTS.md) for the full evidence and limitations.

## Build and reuse

From the repository root, use the project environment and a TeX distribution with `pdflatex` and `bibtex` on `PATH`:

```powershell
an/.venv/Scripts/python.exe an/paper_cvpr2026/build.py
```

The helper requires `pypdf`. It runs pdfLaTeX, BibTeX, then pdfLaTeX twice, writes the final PDF under `output/pdf/`, and places temporary files and optional page previews in `build/`. The build directory can be deleted and regenerated.

The supplied table and figure are sufficient to compile the paper. To regenerate them from the frozen results first, run:

```powershell
an/.venv/Scripts/python.exe an/paper_cvpr2026/make_assets.py
```

Asset generation reads `../results/main/metrics.csv` and `../baseline_comparison/run1024/comparison_summary.json`. Preserve those relative paths when exporting the reporting project. A standalone source bundle can instead be compiled directly from `main.tex` with pdfLaTeX and BibTeX; Python is unnecessary when using the supplied assets.

To integrate the subsection into a larger paper, copy `subsection.tex`, `processing_table.tex`, and the vector figure, then merge `references.bib`. Set `\graphicspath{{figures/}}` and place `\input{subsection}` within Experiments. Figure and table numbers follow the surrounding document. Adjust float placement as needed; the `samepage` group around the implementation-comparison paragraph may be removed.

## Provenance

`template_provenance.json` and `latex_dependencies.json` identify the template and vendored packages. `data_provenance.json` identifies the numerical inputs. `artifact_verification.json` is the historical verification of the 9 November 2026 paper revision, including its source hashes and layout checks. Later documentation and code cleanup is recorded separately; that historical inventory should not be interpreted as a hash inventory of the current directory.

The numerical evidence is retained in [main-study results](../results/main/) and [matched implementation results](../baseline_comparison/run1024/). The paper scripts read those artifacts without rerunning inference.

- [Official CVPR 2026 author guidelines](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines)
- [Pinned CVPR 2026 template](https://github.com/cvpr-org/author-kit/tree/12909ae437f6dbc7435069cfdb4ca44c18e6a02f)
- [FSD paper](https://arxiv.org/abs/2503.21003)
- [COCO preprint](https://arxiv.org/abs/1405.0312)
- [Synthbuster dataset](https://zenodo.org/records/10066460)
