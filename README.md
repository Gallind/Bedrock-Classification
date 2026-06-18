# Seabed Classification of Multibeam Echosounder (MBES) Data

**Per-pixel classification of the Israeli continental shelf into _rock_, _shallow rock_, and _sand_ from multibeam echosounder surveys.**

A reproducible pipeline that turns raw acoustic survey data from the R/V *Bat-Galim*
(Israel Oceanographic & Limnological Research, IOLR) into georeferenced training tiles
and trains seabed classifiers on them. It extends the binary bedrock / non-bedrock U-Net
of Garone et al. (*Frontiers in Earth Science*, 2023 — `DL_article.pdf`) to **three
classes** on Israeli surveys, and pairs the deep model with an interpretable per-pixel
tree baseline for an honest, data-efficient comparison.

> Reichman University — "Code4Good" course project, in collaboration with IOLR.
>
> A short narrated video walkthrough of the whole pipeline lives in [`demo/`](demo/)
> (rendered to `demo/out/seabed-demo.mp4`).

---

## Abstract

Mapping the seabed into substrate classes is a labor-intensive manual task for marine
geologists. This project automates it for IOLR surveys. We build a deterministic tiling
pipeline (`seabed_tiler`) that co-registers bathymetry, backscatter, and slope rasters
onto a common 1 m UTM grid, rasterizes expert polygon annotations into class labels, and
emits overlapping feature/label GeoTIFF tile pairs. On this dataset we train two model
families: a lightweight **U-Net** (`seabed_unet`) that learns spatial context, and an
interpretable **per-pixel tree baseline** (`seabed_forest`, Random Forest +
HistGradientBoosting) optionally refined with an edge-aware **guided-filter
regularization** of the posterior. On a within-survey development split the 3-band U-Net
reaches macro-Dice **0.784**; on the honest leave-one-polygon-out (cross-survey) protocol
it reaches **0.608 ± 0.084**. The dominant and persistent error is *shallow rock* ↔ *sand*
confusion; *rock* is recovered robustly everywhere. The principal limiter is **labeled
area** — roughly 1–2 km² across the surveys here, versus 576 km² in the reference paper —
so the highest-value upgrade is more annotated data, not a different architecture.

---

## Background

**Multibeam echosounders** insonify the seabed across a wide swath and record, per ping,
both **bathymetry** (depth) and **backscatter** (acoustic reflectivity, a proxy for
substrate hardness/roughness). Combined with derived **slope**, these layers carry most of
the signal a geologist uses to distinguish exposed rock from sediment. Automating the
substrate classification frees expert time and produces consistent, repeatable maps.

The reference work — **Garone et al. (2023)** — trains a U-Net to segment MBES mosaics into
bedrock vs. non-bedrock over a large (576 km²) annotated area. This project differs in three
ways that matter:

- **Three classes** (`rock`, `shallow_rock`, `sand`) rather than binary bedrock, where
  `shallow_rock` is partly *depth-defined* and therefore the hardest class.
- **Multiple small surveys** with a real domain shift between them (backscatter is stored as
  JPEG grayscale on some polygons, dB on others) — handled by per-survey self-normalization.
- **Two orders of magnitude less labeled area**, which makes cross-survey generalization, not
  in-survey accuracy, the meaningful benchmark.

---

## Pipeline at a glance

```
DataBase/                      raw, read-only survey bundles (LFS)
  <polygon>/                   bathymetry/backscatter/slope (.xyz/.jpg) + label shapefiles
        │
        ▼  seabed_tiler  (align → grid → labels → tiler)
outputs/<polygon>/<run_tag>/   co-registered feature/label GeoTIFF tile pairs (+ manifest)
        │
        ├──▼  seabed_unet      U-Net (CE+Dice, spatial context)
        └──▼  seabed_forest    RF / HGB per-pixel  ( + guided-spatial regularization )
                  │
                  ▼
training/runs/<experiment>/    checkpoints, metrics, classified maps
reports/                       publication figures + final classified maps + watch GIFs
```

---

## Data & annotation

`DataBase/` holds the raw survey polygons. Each polygon has co-located layers, all on the
same survey footprint:

| Layer | Format | Role |
|---|---|---|
| Bathymetry | `.xyz` (X Y Z) / `.jpg` render | depth — primary feature, also the guided-filter guide |
| Backscatter | `.xyz` / `.jpg` render | acoustic reflectivity — substrate hardness proxy |
| Slope | `.xyz` | terrain steepness — derived feature |
| Labels | shapefile(s) | expert-annotated seabed-class polygons |

**Classes:** `rock = 1`, `shallow_rock = 2`, `sand = 3`, `background = 0` (unlabeled — not a
class; ignored in loss and metrics). Polygons **1, 3, 4, 5** are annotated and used by the
U-Net; the tree baseline additionally trains on **polygon 6**.

**Storage policy.** Only raw `DataBase/` inputs are versioned, via **Git LFS** (`.xyz`,
`.tif`, `.jpg`). Generated `outputs/` and `training/runs/` are deterministic and
**git-ignored** — every contributor regenerates them by re-running the pipeline. Run
`git lfs pull` after cloning to materialize the raw data (see [Reproducing](#reproducing)).

---

## Methods

### Tiling (`seabed_tiler`)

A pure, config-driven pipeline (no install step; `PYTHONPATH=tiling/src`):

- **`align.py`** — computes the intersection extent of all layers, snaps the origin,
  reprojects everything to **EPSG:32636** (UTM 36N) at 1 m, returns aligned float32 arrays.
- **`grid.py`** — I/O-free `build_windows()`; **128 m** tiles at **50 % overlap**
  (`stride = size·(1−overlap)`), producing run-tag `t128m_o50pct_r1m`. A `_rot` variant
  aligns tiles to the annotation minimum-bounding-rectangle; `_rotaug` adds rigid D4
  augmentation passes (**train-only**).
- **`labels.py`** — two rasterization strategies: `shapefile` (one file, class from a NAME
  field via ordered regex) and `shapefile_per_class` (one file per class, burned in priority
  order so rock wins on overlap; `polygonize`/`close_tolerance_m` recover annotator-left-open
  rings — polygon3 needed this).
- **`tiler.py`** — filters windows by `min_valid_frac` / `require_label` and writes
  co-registered `features/*.tif` (multiband float32) + `labels/*.tif` (uint8) + manifests.

A pixel is **valid** only where *every* feature band has real data (`feature_nodata =
-9999.0`); `label_nodata = 0`.

### U-Net (`seabed_unet`)

Classic U-Net, depth 4, base 16 filters (~1.9 M params), trained from scratch (bands are
physical measurements, not RGB — no useful pretrained weights). Variable input channels
(**3-band** = backscatter + bathymetry + slope, or **2-band** = bathymetry + slope).
**Masked weighted CE + soft-Dice** loss (class weights = inverse train-pixel frequency);
AdamW with early stopping on validation macro-Dice; seeded for determinism.

### Per-pixel tree baseline (`seabed_forest`)

A scikit-learn baseline on the **same 3 bands**: **Random Forest** (300 trees) and
**HistGradientBoosting**, CPU-only, reusing `seabed_unet`'s data / normalization / metrics.
Each valid pixel is one `(backscatter, bathymetry, slope)` sample; train pixels are deduped
to undo the 50 % tile overlap. Trains on base `_rot` tiles only (augmentation is a no-op for
a context-free model) and includes polygon6, so it sees five surveys.

### Guided-spatial regularization

A per-pixel classifier has no spatial context. `seabed_forest.spatial` smooths the
**assembled posterior** (the cross-tile-blended probability field, before argmax) with an
edge-aware **guided filter** (He, Sun & Tang, 2013) guided by the bathymetry band —
removing salt-and-pepper while preserving depth boundaries. It is *not* a separate model;
pure numpy/scipy, no new dependencies (radius 4, eps 1e-3).

### Evaluation protocol & normalization

Random tile splits are **forbidden** — 50 %-overlapping tiles leak near-duplicate pixels.
Two whole-polygon spatial protocols are used (per `docs/DATA_AUGMENTATION.md`, the binding
contract):

- **`spatial_blocks`** (development) — each survey is cut into contiguous VAL / TRAIN / TEST
  bands along its long axis, with 96 m buffers (> tile half-diagonal) so splits share **zero
  pixels**. All surveys contribute to training.
- **`polygon` / LOPO** (reporting) — leave-one-polygon-out cross-validation; the only number
  that measures generalization to an *unseen* survey.

**Per-band normalization** bridges the domain shift: bathymetry and slope normalize
*globally* (preserving absolute depth — `shallow_rock` is literally depth-defined), while
backscatter normalizes *per-polygon* (self-normalizing each survey across the JPEG↔dB shift;
feature-only, so valid at inference on a new survey).

---

## Results

> **Labels & retraining.** All models — the forest baselines *and* the U-Net — are trained
> on the *corrected* polygon1 labels (commit `a4aeeb5`: priority burn order + recovered the
> mislabeled rock feature). The U-Net was fully retrained for this fix on a CUDA box (RTX
> 4050; see [`docs/UNET_RETRAIN_RTX4050.md`](docs/UNET_RETRAIN_RTX4050.md)), so the metrics,
> figures, classified maps, and watch GIFs below all reflect the retrained models. Numbers
> are quoted verbatim from [`reports/README.md`](reports/README.md) and
> [`training/README.md`](training/README.md).

### Development split — within-survey (spatial blocks, test tile-pixels)

| Model | macro-Dice | overall acc. | Cohen's κ | rock | shallow_rock | sand |
|---|---|---|---|---|---|---|
| RF (3-band) | 0.716 | 0.741 | 0.502 | 0.900 | 0.445 | 0.802 |
| HGB (3-band) | 0.728 | 0.764 | 0.531 | 0.906 | 0.456 | 0.823 |
| **U-Net (3-band)** | **0.784** | **0.782** | **0.599** | **0.955** | **0.568** | **0.827** |
| U-Net (2-band) | 0.670 | 0.670 | 0.386 | — | — | — |

### Cross-survey generalization — leave-one-polygon-out (mean ± std, 4 folds)

| Model | macro-Dice | overall acc. | rock | shallow_rock | sand |
|---|---|---|---|---|---|
| **U-Net (3-band)** | **0.608 ± 0.084** | **0.710 ± 0.121** | **0.841** | **0.371** | **0.612** |
| U-Net (2-band) | 0.483 ± 0.155 | 0.582 ± 0.188 | 0.629 | 0.239 | 0.580 |
| RF (3-band) | 0.537 ± 0.108 | 0.592 ± 0.176 | 0.779 | 0.313 | 0.518 |
| HGB (3-band) | 0.537 ± 0.118 | 0.598 ± 0.188 | 0.793 | 0.302 | 0.515 |

### Guided-spatial regularization (full-polygon map, raw → regularized)

| Protocol | Model | macro-Dice | shallow_rock Dice |
|---|---|---|---|
| dev (blocks) | RF | 0.728 → **0.739** | 0.465 → 0.493 |
| dev (blocks) | HGB | 0.733 → **0.743** | 0.465 → 0.488 |
| LOPO | RF | 0.542 → **0.559** | 0.311 → 0.314 |
| LOPO | HGB | 0.541 → **0.547** | 0.301 → 0.295 |

### Figures

Learning curves (train loss + val/test macro-Dice per family):

![Learning curves](reports/learning_curves.png)

Row-normalized confusion matrices on the test split (diagonal = recall):

![Confusion matrices](reports/confusion_matrices.png)

Aggregate + per-class metrics and the forest raw→spatial effect:

![Metrics by type](reports/metrics_by_type.png)

Example classified maps for **polygon1** (red = rock, salmon = shallow_rock, blue = sand;
grey = unlabeled/no-coverage):

| Ground truth | U-Net (3-band) | RF + guided-spatial |
|:---:|:---:|:---:|
| ![truth](reports/classified_maps/polygon1/polygon1__ground_truth__t128m_o50pct_r1m.png) | ![unet](reports/classified_maps/polygon1/polygon1__unet__experiment_3band__t128m_o50pct_r1m.png) | ![rf-spatial](reports/classified_maps/polygon1/polygon1__random_forest_spatial__forest_3band__t128m_o50pct_r1m.png) |

Live multi-model viewer (`seabed_forest.watch`) on **polygon3** — top row is the current
tile's input bands and each family's per-tile prediction; the grid below is each family's
full-polygon map filling in tile by tile (RF, HGB, their guided-spatial variants, U-Net,
and ground truth):

<img src="reports/watch_gifs/polygon3_watch_multi.gif" alt="Multi-model watch — polygon3" width="800">


The remaining per-polygon animations live in
[`reports/watch_gifs/`](reports/watch_gifs/); the full per-model map set is in
[`reports/classified_maps/`](reports/classified_maps/). See
[`reports/README.md`](reports/README.md) for the complete report.

---

## Discussion & limitations

- **`shallow_rock` ↔ `sand` is the dominant error for every model.** It is the contextual,
  depth-defined class; per-pixel trees collapse on it cross-survey (LOPO Dice ≈ 0.02–0.05 on
  the polygon5/6 folds), and even the U-Net trails here (0.371 LOPO).
- **`rock` is recovered robustly** — recall ≥ 0.89 within-survey and Dice ≥ 0.84 cross-survey
  (U-Net) — so the maps are reliable where it matters most for hazard/habitat work.
- **Backscatter earns its place:** 3-band beats 2-band on every metric under both protocols,
  confirming the per-polygon backscatter normalization bridges the domain shift.
- **Guided-spatial regularization is a free, consistent lift** (dev ≈ +0.01, LOPO RF +0.017)
  driven mostly by `rock` and cleaner maps — but it reinforces existing signal and cannot
  manufacture context the model never captured, so it does not close the gap to the U-Net.
- **The accuracy ceiling is not raised by swapping models** on these three raw bands. The real
  levers are engineered/multi-scale features and, above all, **more annotated area** — the
  ~1–2 km² labeled here is two orders of magnitude below the 576 km² of Garone et al. (2023).
  These runs establish a reproducible baseline and a working pipeline, not a production
  classifier.

---

## Reproducing

**Clone & fetch raw data** (Git LFS required):

```bash
git lfs install
git lfs pull
```

**Two virtual environments** (the tiler and the trainer have incompatible Python needs):

```bash
# Tiler — GDAL stack (Python 3.14 .venv)
.venv/bin/python -m pip install -r tiling/requirements.txt

# Trainer — PyTorch 2.2.2 (Python 3.12 .venv-train; no macOS x86_64 torch wheel past 2.2.x)
python3.12 -m venv .venv-train
.venv-train/bin/python -m pip install -r training/requirements.txt
```

**Generate the tile dataset** (rotation-aware + augmentation passes):

```bash
PYTHONPATH=tiling/src .venv/bin/python -m seabed_tiler --config tiling/config/polygon1.yaml --rotated
PYTHONPATH=tiling/src .venv/bin/python -m seabed_tiler --config tiling/config/polygon1.yaml --augment
```

**Train & evaluate** (run from repo root):

```bash
export PYTHONPATH=tiling/src:training/src

# U-Net
.venv-train/bin/python -m seabed_unet.train    --config training/config/experiment_3band.yaml
.venv-train/bin/python -m seabed_unet.evaluate --config training/config/experiment_3band.yaml
.venv-train/bin/python -m seabed_unet.predict  --config training/config/experiment_3band.yaml --polygon polygon4
.venv-train/bin/python -m seabed_unet.crossval --config training/config/experiment_3band.yaml   # LOPO

# Tree baseline (+ guided-spatial)
.venv-train/bin/python -m seabed_forest.train    --config training/config/forest_3band.yaml
.venv-train/bin/python -m seabed_forest.evaluate --config training/config/forest_3band.yaml
.venv-train/bin/python -m seabed_forest.predict  --config training/config/forest_3band.yaml --polygon polygon4 --spatial
```

**Tests:**

```bash
.venv/bin/python -m pytest tiling/tests -q          # tiling pipeline
.venv-train/bin/python -m pytest training/tests -q  # models
```

Full command references and flags are in [`tiling/README.md`](tiling/README.md) and
[`training/README.md`](training/README.md).

---

## Repository layout

```
DataBase/                raw survey bundles (LFS, read-only)        — versioned
outputs/                 generated tile datasets                   — git-ignored
tiling/                  seabed_tiler package
  src/seabed_tiler/        config · align · grid · labels · tiler
  config/                  default.yaml + per-polygon YAMLs
  tests/
training/                model packages
  src/seabed_unet/         U-Net training / eval / predict / crossval / watch
  src/seabed_forest/       RF + HGB baseline + guided-spatial
  config/                  experiment_{2,3}band.yaml · forest_3band.yaml
  runs/                    checkpoints, metrics, maps               — git-ignored
  tests/
reports/                 publication figures, classified maps, watch GIFs
demo/                    narrated walkthrough video (Remotion, React/TS → MP4)
docs/                     data-augmentation contract & setup guides
DL_article.pdf           reference paper (Garone et al., 2023)
```

---

## Documentation

| Document | Contents |
|---|---|
| [`tiling/README.md`](tiling/README.md) | tiler usage, config tuning, adding a polygon |
| [`training/README.md`](training/README.md) | model design, full results, forest baseline & spatial |
| [`reports/README.md`](reports/README.md) | model-comparison report, figures, classified maps |
| [`docs/DATA_AUGMENTATION.md`](docs/DATA_AUGMENTATION.md) | binding split/augmentation/normalization contract |
| [`docs/TRAINING_DATA_SETUP.md`](docs/TRAINING_DATA_SETUP.md) | regenerating the tile dataset |
| [`docs/UNET_RETRAIN_RTX4050.md`](docs/UNET_RETRAIN_RTX4050.md) | U-Net retrain on the corrected labels |
| [`demo/README.md`](demo/README.md) | narrated Remotion walkthrough video — build & render `out/seabed-demo.mp4` |

---

## References

1. Garone, R. V., et al. (2023). *Seabed classification of multibeam echosounder data into
   bedrock/non-bedrock using deep learning.* **Frontiers in Earth Science** (`DL_article.pdf`).
2. He, K., Sun, J., & Tang, X. (2013). *Guided Image Filtering.* IEEE TPAMI 35(6), 1397–1409.
3. Ronneberger, O., Fischer, P., & Brox, T. (2015). *U-Net: Convolutional Networks for
   Biomedical Image Segmentation.* MICCAI.

---

*Coordinate reference system: EPSG:32636 (UTM Zone 36N, meters) throughout. Built with IOLR
for the Reichman "Code4Good" program.*
