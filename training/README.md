# seabed_unet — U-Net training on the seabed tile dataset

Per-pixel classification of the tiled seabed dataset (rock / shallow_rock / sand)
with a lightweight U-Net trained from scratch. Consumes the rotation-aware runs the
tiler produces under `outputs/` and obeys the binding contract in
`docs/DATA_AUGMENTATION.md` (spatial splits only, augmented tiles train-only,
loss masked by feature validity, rigid D4 augmentation only).

## Environment

This is a **separate venv** from the tiler's: PyTorch dropped macOS x86_64 wheels
after 2.2.x and those only exist up to Python 3.12 (the tiler's `.venv` is 3.14).

```bash
python3.12 -m venv .venv-train
.venv-train/bin/python -m pip install -r training/requirements.txt
```

Prerequisite: the tile dataset must exist — see `docs/TRAINING_DATA_SETUP.md`
(`outputs/<polygon>/t128m_o50pct_r1m_rot` and `_rotaug` for all four polygons).

## Commands (run from repo root)

```bash
export PYTHONPATH=tiling/src:training/src

# Train (writes best.pt, normalization_stats.json, history.csv to training/runs/<name>/)
.venv-train/bin/python -m seabed_unet.train --config training/config/experiment_3band.yaml

# Evaluate on the held-out test polygon (metrics.json + confusion matrix CSV/PNG)
.venv-train/bin/python -m seabed_unet.evaluate --config training/config/experiment_3band.yaml

# Classified map (georeferenced GeoTIFF + colorized JPEG) for any polygon
.venv-train/bin/python -m seabed_unet.predict --config training/config/experiment_3band.yaml --polygon polygon4

# Leave-one-polygon-out cross-validation (4 retrains; the honest cross-survey number)
.venv-train/bin/python -m seabed_unet.crossval --config training/config/experiment_3band.yaml

# Tests
.venv-train/bin/python -m pytest training/tests -q
```

## Design (decisions and why)

- **3-class output** (rock=1, shallow_rock=2, sand=3). Background (0) is *unlabeled*,
  not a class: it is ignored in loss and metrics together with every pixel where any
  feature band is nodata (−9999) / NaN — otherwise the model learns that "no data"
  means "background seabed".
- **Two split schemes** (config `split.mode`). Random tile splits are forbidden —
  tiles overlap 50%, so they leak near-duplicate pixels (`docs/DATA_AUGMENTATION.md`):
  - `spatial_blocks` (default, development): every polygon is cut into contiguous
    VAL | TRAIN | TEST regions along its survey long axis; tiles within `buffer_m`
    (96 m > tile half-diagonal 90.5 m) of a boundary are dropped, so splits share
    **zero pixels** by construction. All four surveys contribute to training.
    The small regions sit at the strip ends because a middle band would need
    2× buffer of clearance — wider than the band itself on these short surveys.
  - `polygon` (reporting): whole-polygon holdout; `seabed_unet.crossval` runs
    leave-one-polygon-out over all four polygons and reports mean±std — the only
    number that measures generalization to an unseen survey.
- **Per-band normalization** (config `normalization.band_modes`): bathymetry and
  slope are normalized **globally** (one train-only range, preserving absolute
  depth across surveys — shallow_rock is literally defined by depth, which
  per-polygon scaling erases); backscatter stays **per-polygon** (the survey
  self-normalizes, bridging the JPEG-grayscale-vs-dB domain shift; feature-only,
  so legitimate at inference on a new survey).
- **Loss** = weighted CE + soft Dice (both masked); class weights = inverse pixel
  frequency from train pixels only.
- **Architecture**: classic U-Net, depth 4, base 16 filters (~1.9M params), variable
  input channels (2 or 3), from scratch — bands are physical measurements, not RGB,
  so there are no meaningful pretrained weights.
- **Determinism**: seeded RNGs, single-process loading; reruns wipe their run dir
  (same policy as the tiler).

## Experiments

Two band configurations, same split, same seed:

| Experiment | Config | Bands |
|---|---|---|
| E1 | `experiment_3band.yaml` | backscatter + bathymetry + slope |
| E2 | `experiment_2band.yaml` | bathymetry + slope |

### Results — round 1 (spatial blocks + per-band normalization, seed 42)

Two protocols, two questions. They are **not comparable to each other**: the
development split tests within surveys the model has partly seen; LOPO tests on
a survey the model has never seen.

#### Development split (spatial blocks; val/test = 14 tiles each)

| Metric (test regions) | 3-band | 2-band |
|---|---|---|
| overall accuracy | **0.875** | 0.748 |
| Cohen's kappa | **0.729** | 0.454 |
| macro Dice | **0.836** | 0.665 |
| rock Dice | **0.956** | 0.878 |
| shallow_rock Dice | **0.642** | 0.294 |
| sand Dice | **0.912** | 0.822 |

#### Cross-survey generalization (LOPO, mean ± std over 4 folds)

| Metric (held-out polygon) | 3-band | 2-band |
|---|---|---|
| overall accuracy | **0.724 ± 0.100** | 0.551 ± 0.227 |
| Cohen's kappa | **0.496 ± 0.133** | 0.284 ± 0.250 |
| macro Dice | **0.644 ± 0.094** | 0.473 ± 0.157 |
| rock Dice | **0.817 ± 0.080** | 0.636 ± 0.200 |
| shallow_rock Dice | **0.419 ± 0.270** | 0.208 ± 0.153 |
| sand Dice | **0.697 ± 0.218** | 0.575 ± 0.251 |

Artifacts: `training/runs/<exp>/eval_test/` (blocks), `training/runs/<exp>_lopo/`
(per-fold runs + `summary.json`), maps in `training/runs/<exp>/maps/`.

**Reading of the results.**
- **Per-band normalization worked.** Round 0 (per-polygon scaling everywhere,
  whole-polygon split test=polygon4) scored macro-Dice 0.496 (3-band); the
  matching LOPO fold now scores **0.618** on the same test polygon, and sand
  Dice on that fold jumped 0.476 → 0.743. Restoring absolute depth was the
  single biggest lever, exactly as hypothesized (shallow_rock is depth-defined).
  (Caveat: round-0 trained on poly1+5/val poly3; the LOPO fold trains on
  poly1+3/val poly5 — same test survey, slightly different train mix.)
- **Backscatter earns its place.** 3-band beats 2-band on every metric under
  both protocols (LOPO macro-Dice 0.644 vs 0.473) — per-polygon backscatter
  normalization successfully bridges the grayscale-vs-dB shift; round 0's
  "neither dominates" verdict is overturned.
- **The predicted polygon4 map** now reproduces the reference structure: rock
  massif, shallow_rock halo on the correct flank, clean sand basin (compare
  `runs/unet_3band/maps/polygon4_pred.jpg` with
  `outputs/polygon4/t128m_o50pct_r1m_rot/stitched/labels.jpg`).
- **Remaining weak spot: shallow_rock across surveys** (LOPO 0.419 ± 0.270; the
  polygon5 fold scores 0.000 — that survey has only ~11% shallow_rock pixels and
  12 base tiles, so the class essentially vanishes from its fold). More
  annotated area with shallow_rock is the highest-value data request.
- The dev-split numbers (0.836 macro-Dice) are the optimistic within-survey view;
  quote the LOPO numbers when describing generalization. Dev val/test are only
  14 tiles each — the geometric maximum for leak-free tile-level eval regions on
  these short survey strips (96 m buffers eat 15% bands; polygon3 contributes
  train-only).

Next probes: hillshade 4th band, D4 test-time augmentation + E3/E4 ensemble
(inverse_op exists in `seabed_tiler.augment`), RF per-pixel baseline.

## Honest expectations

The labeled area (~1–2 km² across four surveys) is two orders of magnitude smaller
than the reference paper's training set (Garone et al. 2023, 576 km²). These runs
establish a reproducible baseline and a working pipeline, not a production
classifier. The highest-value upgrade is new annotated survey polygons, not more
tuning (see "Honest accounting" in `docs/DATA_AUGMENTATION.md`).
