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

# Tests
.venv-train/bin/python -m pytest training/tests -q
```

## Design (decisions and why)

- **3-class output** (rock=1, shallow_rock=2, sand=3). Background (0) is *unlabeled*,
  not a class: it is ignored in loss and metrics together with every pixel where any
  feature band is nodata (−9999) / NaN — otherwise the model learns that "no data"
  means "background seabed".
- **Split is by whole polygon** (config `split:`): test = polygon4, val = polygon3,
  train = polygon1 + polygon5 (`_rot` + `_rotaug` + training-time D4 via
  `seabed_tiler.augment`). Tiles overlap 50%, so random tile splits would leak.
  Future polygons: add them to the split lists; full leave-one-polygon-out CV is a
  config change.
- **Per-polygon normalization** (default): each survey self-normalized to [0,1] with
  2–98 percentile clipping, computed from features only — legitimate at inference
  (a new survey self-normalizes) and required to bridge the backscatter domain shift
  (polygon1 = JPEG grayscale 0–255 vs polygons 3/4/5 = real dB). `global` mode
  (train-stats only) is available for paper-faithful comparison.
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

### Results (test = polygon4, never seen in training)

Both runs: seed 42, early stopping on val (polygon3) macro-Dice, ~10 min on MPS.

| Metric (test) | E1 3-band | E2 2-band |
|---|---|---|
| best val macro-Dice (polygon3) | **0.807** (ep 20/45) | 0.791 (ep 9/34) |
| overall accuracy | 0.448 | **0.480** |
| Cohen's kappa | 0.110 | **0.250** |
| macro Dice | **0.496** | 0.461 |
| rock Dice (PAcc/UAcc) | **0.736** (0.71/0.77) | 0.430 (0.77/0.30) |
| shallow_rock Dice | 0.277 (0.24/0.33) | **0.584** (0.62/0.56) |
| sand Dice | **0.476** (0.55/0.42) | 0.369 (0.23/0.87) |

Reports: `training/runs/<exp>/eval_test/metrics.json` + confusion matrices;
maps: `training/runs/<exp>/maps/polygon4_pred.{tif,jpg}` (compare against
`outputs/polygon4/t128m_o50pct_r1m_rot/stitched/labels.jpg`).

**Reading of the results.** The val→test drop (0.80 → ~0.47 macro-Dice) is the
cross-survey generalization gap expected from ~1–2 km² of training data — val
scores look good because polygon3 resembles the training surveys more than
polygon4 does. The two band sets fail differently: **E1 (with backscatter)
recovers the rock outcrops well** — its predicted map places the central rock
massif correctly — but collapses most of the shallow_rock halo into sand.
**E2 (bathy+slope only) trades that for shallow_rock recall**, over-painting
rock across the survey (rock UAcc 0.30). Neither dominates; with whole-polygon
holdouts this small, per-fold variance is large, so treat these as a baseline
to beat, not a model selection verdict. Next probes: full 4-fold LOPO CV (config
change only) and a hillshade band.

## Honest expectations

The labeled area (~1–2 km² across four surveys) is two orders of magnitude smaller
than the reference paper's training set (Garone et al. 2023, 576 km²). These runs
establish a reproducible baseline and a working pipeline, not a production
classifier. The highest-value upgrade is new annotated survey polygons, not more
tuning (see "Honest accounting" in `docs/DATA_AUGMENTATION.md`).
