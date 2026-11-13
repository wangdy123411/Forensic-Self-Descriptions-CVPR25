# Maintenance and report packaging

The 13 November 2026 cleanup improves navigation, prose, and code readability.
It does not change the completed study's predictions, features, sample selection,
weights, metrics, figures, or paper text. Source edits and cleanup remain inside `an/`.

## Current material

Start with the [report collection](../reports/index.html). It links the current
PDF and LaTeX source to the main study, matched implementation comparison, and
methods. `an/export_reports.py` creates `an/reports/AN_Robustness.zip` with the
reports, figures, tables, provenance, and maintained source. The ZIP includes a
SHA-256 manifest and a local `START_HERE.html` entry point.

The portable package omits original image payloads, model weights, the Python
environment, and per-image feature arrays. These remain in the working repository
and are needed to rerun inference or the full feature audit. The supplied paper
assets compile independently of those large files.

## Source maintenance

Experiment code uses consistent formatting and clearer function descriptions.
The computational code's executable syntax tree is unchanged after excluding
docstrings. Verification tools can now write separate audit outputs rather than
overwrite the historical study records.

The comparison runner retains strict source-hash and resume checks. An explicit
`--historical-source-archive` option permits a fresh comparison with stored
features only after the archived source matches the original run hashes and the
current executable syntax tree matches that source. This permits formatting and
docstring edits; it does not permit changes to inference logic. See the
[comparison instructions](../baseline_comparison/README.md).

## Retained history

`provenance/pre_cleanup_source.zip` contains original scoring and analysis source
files and revised documentation. Its three scoring files match the hashes in
both frozen phase signatures. This is a targeted archive, not a complete prior
checkout. `provenance/paper_history.zip` preserves earlier paper drafts and the
previous source package.

`provenance/maintenance_records.zip` retains the detailed cleanup checks and
temporary source backups. Every archived file was verified before the temporary
workspace was removed.

Historical provenance JSON files describe earlier file states. They are retained
without rewriting their claims. New checks are recorded in
`provenance/cleanup_validation.json` and
`provenance/cleanup_result_verification.json`.

Temporary paper builds, synthetic rendering fixtures, bytecode, stale stop
markers, cached installers, and three completed-run controller scripts were
removed. `provenance/cleanup_removals.json` records these removals. Reusable
acquisition, scoring, analysis, monitoring, and validation tools remain.
