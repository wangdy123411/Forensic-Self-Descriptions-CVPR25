# Dataset access and subset protocol

The study uses native image files from public archives. Selection is
deterministic and independent of detector outputs. These local subsets do not
reconstruct the FSD paper's unpublished image-ID splits.

## Scope of the executed study

The chosen study contains **400 original images**: 100 COCO2017 calibration
photographs, 100 disjoint COCO2017 test photographs, and 50 test images from each
of four Synthbuster sources (DALL-E 3, Adobe Firefly, Stable Diffusion XL, GLIDE).
Thus task metrics use 300 test originals; the 100 calibration originals are
reserved for threshold selection. All processed variants retain the original's
partition. RAISE is excluded from this study.

## Why these datasets

- **Synthbuster** is explicitly one of the synthetic collections pooled by the
  FSD paper. Its official archive has 1,000 images per generator for nine
  generators, and the curator states that no JPEG compression or resampling was
  added to those images. This describes curator-added degradation, not a
  guarantee that every payload is lossless: our downloaded Firefly members have
  `.png` names but JPEG payloads, confirmed by Pillow and JPEG magic bytes.
  We select DALL-E 3, Adobe Firefly, Stable Diffusion XL,
  and GLIDE. The first three names correspond to sources supported by the
  released attribution models; GLIDE is an intentionally unseen source for those
  released models. Source identity is read from archive folders, never inferred
  from detector predictions. See the [official Synthbuster dataset
  record](https://zenodo.org/records/10066460).
- **COCO 2017** is a real-image domain used in FSD's experiments. The separate
  `acquire_coco.py` downloader uses the official validation archive. The exact
  image membership of the released detector's training data is unavailable, so
  training overlap cannot be independently ruled out. JPEG originals also
  introduce a format/domain difference relative to Synthbuster PNGs.
- **RAISE-1k was attempted but not used.** It is a real-image control suggested by the Synthbuster
  curator, because its image descriptions supplied the synthetic prompts.
  RAISE is **not** one of the four real-image domains listed in the FSD paper.
  Adding it extends the test rather than reproducing the paper's real-data
  protocol. The public mirror URL used by the associated detection project is
  `https://www.grip.unina.it/download/prog/DMimageDetection/real_RAISE_1k.zip`;
  this mirror is explicitly listed in the [author-maintained Synthbuster
  download script](https://github.com/grip-unina/ClipBased-SyntheticImageDetection/blob/main/data/synthbuster_download.sh).
  The [original RAISE collection page](http://mmlab.science.unitn.it/RAISE/)
  identifies the source collection. The download script requests noncommercial
  use and citation of the RAISE and Synthbuster papers.

RAISE access attempts on 8 November 2026 failed because of an expired HTTPS
certificate; the HTTP response then raised `zipfile.BadZipFile`. Its response
body was not retained, so the archive failure cannot be diagnosed further.
Certificate verification remained enabled. No RAISE image entered a frozen
manifest, and RAISE is not needed to reproduce the completed study.

The FSD paper's four real domains are COCO2017, ImageNet-1k, ImageNet-22k, and
MIDB; synthetic datasets are pooled from OSSIA, DMID, Synthbuster, and additional
author generations. The synthetic “train” counts in its supplementary data
table apply to source-related experiments, not synthetic training of its
zero-shot detector. See [FSD sections 5.2–5.5 and appendix
A](https://arxiv.org/html/2503.21003v1).

## Selection and provenance

Run from the repository root in the dedicated environment:

```powershell
an/.venv/Scripts/python.exe an/acquire_coco.py --test-count 200 --calibration-count 100 --seed 20261108
an/.venv/Scripts/python.exe an/download_data.py --dataset synthbuster --per-source 50 --seed 20261108
```

The COCO acquisition pool contains 300 photographs, of which the frozen study
selects 200. The Synthbuster target is 200 images. Sample counts were fixed before
scoring, preserving the seeded ordering of the acquisition pool.

`--list-only` downloads ZIP metadata and records archive folders without image
payloads. The default selection seed is `20261108`. Each source's members are
ranked by SHA-256 of `seed|archive_member`; the requested prefix is selected.
Increasing sample counts therefore extends each source's existing subset.
Synthbuster images are test-only. COCO selection uses SHA-256 of
`seed:archive_member`, with the first 100 selected originals assigned to
calibration and subsequent originals to test. These are our experiment splits,
not author-provided official partitions. The downloader also retains optional
RAISE acquisition support for future use, but it is not part of this run.

The downloader makes bounded HTTP range requests, requires status 206 and exact
Content-Range agreement, and checks each selected member's ZIP CRC-32. It writes
original file bytes without recompression, computes SHA-256, verifies image
decoding, and records native dimensions. Existing files are checked against the
archive before being reused. The manifest is updated atomically after every
successful image, so a partial download is inspectable and resumable.

Outputs live under `an/data/`: `manifest_synthbuster.csv`, `manifest_coco.csv`,
folder inventories, download provenance JSON, and `images/` containing originals.
Acquisition manifests contain `image_id`, repository-relative `path`, binary `label`
(0 real, 1 synthetic), `source`, `split`, SHA-256, dimensions, source URL,
dataset, original member name, CRC-32, and selection seed.

Freeze a ready subset with native format inspection and complete image/hash
verification:

```powershell
an/.venv/Scripts/python.exe an/prepare_manifest.py --per-source 0 --coco-test 100 --coco-calibration 100 --raise-test 0 --raise-calibration 0 --output an/data/manifest_real.csv
an/.venv/Scripts/python.exe an/prepare_manifest.py --per-source 50 --coco-test 0 --coco-calibration 0 --raise-test 0 --raise-calibration 0 --output an/data/manifest_fake.csv
```

Each phase manifest contains 200 originals. Both use the same frozen processing
protocol and are combined only after scoring, allowing acquisition and extraction
to proceed independently.

The preparer refuses incomplete requested source/split counts and duplicate
image IDs, paths, or SHA-256 values. It adds actual file `format` from Pillow,
which is essential for detecting the Firefly extension/payload mismatch, and
renames the RAISE real domain to `RAISE-1k`. Zero counts skip a domain entirely.
It preserves the acquisition's seeded ordering and existing split assignments.
A frozen output cannot be changed without an explicit `--force` flag; identical
reruns are allowed. The companion `.provenance.json` records frozen manifest and
input-manifest hashes plus exact source/split/format counts.

## Archive compatibility checks

`an/tests/test_data_acquisition.py` runs offline against simulated HTTP range
responses. It checks strict status/range validation, truncated bodies, ordinary
ZIP extraction and CRC rejection, forced ZIP64 local headers, and a sparse ZIP64
fixture whose member and central-directory offsets exceed 4 GiB. No large test
file is allocated or downloaded. A separate fixture demonstrates that a server
can honor byte ranges for an HTML document while `ZipFile` correctly rejects it;
valid HTTP ranges alone do not prove the response is a ZIP archive.

The successfully indexed official Synthbuster archive is 12,372,557,226 bytes
and contains 9,000 images. Together with the offline tests, this supports ZIP64
compatibility, but it does not diagnose the unretained RAISE HTTP response.

## Interpretation limits

All variants stay in their original's partition. The pretrained model remains
fixed, and only separate real calibration images select the optional threshold.
An evaluation across these limited sources is a robustness study, not a
reproduction of the paper's 24-generator, 96 real-versus-synthetic comparisons.
The source GMMs are frozen: possible overlap between these public datasets and
unpublished release training IDs is unknown and must be reported for attribution.
The 200 selected synthetic files have 181 distinct archive-member stems; 18
stems occur across multiple generators. These are distinct image bytes, but
may share reference content. File-hash uniqueness therefore establishes absence
of exact duplicates, not semantic independence. The stem audit is recorded in
`an/data/shared_stem_audit.json`; content-group sensitivity analysis complements
the primary paired bootstrap over original image IDs.
Report each real domain separately and inspect pooled results for format/domain
confounding. Paired within-image changes under processing are the most direct
evidence about robustness in this setup.
