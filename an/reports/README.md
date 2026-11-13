# AN report hub

Open [index.html](index.html) for the local report index. Start with the
[CVPR subsection PDF](../paper_cvpr2026/output/pdf/robustness_subsection.pdf),
then use the detailed reports for methods, uncertainty, and interpretation.

- [Subsection LaTeX](../paper_cvpr2026/subsection.tex) and
  [complete LaTeX package](../paper_cvpr2026/output/robustness_latex_source.zip).
- [Robustness findings](../RESULTS.md),
  [full report](../results/main/report.md), and
  [experiment protocol](../docs/experiment_plan.md).
- [Original-code comparison](../baseline_comparison/RESULTS.md) and
  [full comparison report](../baseline_comparison/run1024/report.md).
- [Detection metrics](../results/main/metrics.csv),
  [uncertainty intervals](../results/main/auroc_intervals.csv), and
  [generator comparison](../results/main/plots/source_auroc_heatmap.png).

The robustness study evaluates 300 test originals under 19 conditions, with
100 additional clean calibration originals. The implementation validation uses
25 originals under seven conditions. These cohorts answer different questions:
processing sensitivity and implementation agreement. The optimized extractor
improves efficiency on the tested hardware; no robustness mitigation was tested.

`AN_Robustness.zip` is the portable report, evidence, and source bundle in this
working folder. Extract it and open `START_HERE.html` to access the report index
with working relative links.

The bundle excludes raw images, model weights, the virtual environment, and
per-image arrays. Full inference requires the working repository and those
assets; see the [reproduction guide](../README.md). Reports and publication assets
can be read without running inference or accessing the network.
