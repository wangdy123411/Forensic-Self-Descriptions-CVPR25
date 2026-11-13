# Supplementary content-proxy sensitivity

Status: **complete_for_expected_test_pairs**.

The clean fake cohort contains 200 distinct original images and 181 proxy clusters. 18 stems are shared across generators (37 images).

Archive stems may identify shared reference content, but they are not verified prompt IDs. Distinct image bytes do not establish independent content. This is a dependence-assumption sensitivity analysis, not a replacement for the main source-stratified bootstrap.

Each of 1000 draws resamples fake proxy clusters with replacement and assigns the same multiplicity to every image in the selected cluster. Real originals are resampled within each real source. Clean and processed scores reuse identical weights; generator AUROCs are averaged equally. Fake counts per generator vary in this supplement, unlike the main bootstrap. Differences between intervals therefore reflect the resampling-design change as well as potential cross-generator dependence. A missing generator makes the macro draw invalid; the number of valid draws is disclosed. Intervals are pointwise 95% percentiles, without multiple-comparison adjustment.

No real-to-fake content correspondence is inferred or modeled. If a future real-image cohort contains reference images used to generate the synthetic cohort, those relationships require verified identifiers and a joint real/fake cluster design.

Point estimates retain the main image-based, equally weighted generator estimand; no scores, thresholds, checkpoint parameters, or main confidence intervals are changed.

| Condition | Macro AUROC | Paired change | Main lower | Main upper | Proxy-cluster lower | Proxy-cluster upper | Valid draws |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clean | 0.995 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1000 |
| jpeg_q95 | 0.995 | -0.000 | -0.002 | 0.002 | -0.002 | 0.002 | 1000 |
| jpeg_q75 | 0.989 | -0.007 | -0.017 | 0.002 | -0.016 | 0.001 | 1000 |
| jpeg_q50 | 0.967 | -0.028 | -0.047 | -0.013 | -0.046 | -0.013 | 1000 |
| resize_075 | 0.782 | -0.214 | -0.263 | -0.165 | -0.259 | -0.167 | 1000 |
| resize_050 | 0.590 | -0.406 | -0.438 | -0.377 | -0.436 | -0.376 | 1000 |
| resize_025 | 0.848 | -0.148 | -0.192 | -0.107 | -0.192 | -0.107 | 1000 |
| crop_area075 | 0.757 | -0.239 | -0.280 | -0.196 | -0.281 | -0.197 | 1000 |
| crop_area050 | 0.620 | -0.376 | -0.431 | -0.320 | -0.430 | -0.322 | 1000 |
| crop_area025 | 0.592 | -0.404 | -0.433 | -0.374 | -0.433 | -0.372 | 1000 |
| blur_sigma05 | 0.997 | 0.001 | -0.003 | 0.006 | -0.003 | 0.006 | 1000 |
| blur_sigma10 | 0.977 | -0.018 | -0.033 | -0.005 | -0.034 | -0.005 | 1000 |
| blur_sigma20 | 0.992 | -0.004 | -0.014 | 0.006 | -0.014 | 0.006 | 1000 |
| sharpen_050 | 0.995 | -0.000 | -0.001 | 0.001 | -0.001 | 0.001 | 1000 |
| sharpen_100 | 0.995 | -0.000 | -0.002 | 0.001 | -0.002 | 0.001 | 1000 |
| sharpen_200 | 0.995 | 0.000 | -0.002 | 0.002 | -0.002 | 0.002 | 1000 |
| resize075_jpeg75 | 0.788 | -0.207 | -0.260 | -0.158 | -0.262 | -0.156 | 1000 |
| resize050_jpeg75 | 0.618 | -0.378 | -0.403 | -0.349 | -0.408 | -0.348 | 1000 |
| resize050_jpeg50 | 0.631 | -0.365 | -0.397 | -0.334 | -0.395 | -0.336 | 1000 |

[Per-generator, pooled, and macro sensitivity metrics](content_sensitivity_metrics.csv) | [Cluster definitions, counts, coverage, and reproducibility](content_sensitivity_summary.json)

Interpret interval-width changes descriptively. The proxy can miss semantically related images with different stems, and identical stems are not confirmed identical prompts. A stronger future study would sample verified reference-content or prompt IDs jointly across generators, then bootstrap those verified groups.
