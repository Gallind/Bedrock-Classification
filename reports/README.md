# Model comparison report — RF / HGB (+ guided-spatial) vs U-Net

Presentation artifacts comparing the per-pixel tree baselines and the U-Net on all
four annotated survey polygons (plus forest-only `polygon6`).

> **Label-fix caveat.** The **forest** models (RandomForest, HistGradientBoosting and
> their guided-spatial regularization) were **retrained on the corrected polygon1
> labels** (commit `a4aeeb5`: priority burn order + recovered the mislabeled rock
> feature). The **U-Net is the *previous* training on the *old* labels** — it has not
> yet been retrained. Treat U-Net numbers/maps here as the pre-fix reference; the
> retrain runs on the RTX 4050 box (see `docs/UNET_RETRAIN_RTX4050.md`).

## Configs

| Family | Config | Notes |
|---|---|---|
| RF / HGB / spatial | `training/config/forest_3band.yaml` | RF 300 trees (`n_jobs=-1`); HGB `max_iter=300`, early-stop; guided-spatial radius 4, eps 1e-3, guide=bathymetry |
| U-Net 3-band | `training/config/experiment_3band.yaml` | bands: backscatter, bathymetry, slope |
| U-Net 2-band | `training/config/experiment_2band.yaml` | bands: backscatter, bathymetry |

All models: classes `rock=1 / shallow_rock=2 / sand=3`, tiles `t128m_o50pct_r1m`
(128 m tiles, 50 % overlap, 1 m grid), per-polygon self-normalization, whole-polygon
spatial splits.

## Contents

- **`classified_maps/<polygon>/`** — final full-polygon classification, **one image per
  model**, named `<polygon>__<model>__<config>__<run_tag>.png`:
  - `<polygon>__random_forest_raw__forest_3band__…png`
  - `<polygon>__random_forest_spatial__forest_3band__…png`
  - `<polygon>__hist_gradient_boosting_raw__forest_3band__…png`
  - `<polygon>__hist_gradient_boosting_spatial__forest_3band__…png`
  - `<polygon>__unet__experiment_3band__…png` *(previous training, old labels)*
  - `<polygon>__ground_truth__…png`

  Each is the model's class map overlaid on the survey hillshade backdrop (grey =
  unlabeled/no-coverage), rendered with the same colors/normalization as the live viewer.
  Colors: **red = rock**, **salmon = shallow_rock**, **blue = sand**. The config and tile
  run-tag are encoded in every filename.
- **`watch_gifs/<polygon>_watch_multi.gif`** — the live tile-by-tile viewer animation
  (`seabed_forest.watch`): top row = the current tile's input bands + each family's
  per-tile classification; bottom grid = each family's full-polygon map filling in live,
  tree spatial maps re-regularized every tile.
- **`learning_curves.png`** — training curves, train/val/test where the family provides
  them: U-Net (train loss + val macro-Dice + test marker, per epoch); HGB (train/val
  boosting loss per iteration + early-stop); RF is a bagging ensemble (non-iterative, no
  loss curve) so its panel shows the cross-model test macro-Dice for context.
- **`confusion_matrices.png`** — row-normalized (recall on the diagonal) confusion
  matrices on the test split for RF, HGB, U-Net 3-band, U-Net 2-band.
- **`metrics_by_type.png`** — aggregate metrics (macro-Dice / overall accuracy / Cohen's
  kappa), per-class Dice, and the forest map raw→guided-spatial effect.

## Headline metrics (test split, tile-pixel)

| model | macro-Dice | OA | kappa | Dice rock | Dice shallow_rock | Dice sand |
|---|---|---|---|---|---|---|
| RF (3band) | 0.716 | 0.741 | 0.502 | 0.900 | 0.445 | 0.802 |
| HGB (3band) | 0.728 | 0.764 | 0.531 | 0.906 | 0.456 | 0.823 |
| U-Net (3band) *(prev. labels)* | 0.784 | 0.782 | 0.599 | 0.955 | 0.569 | 0.829 |
| U-Net (2band) *(prev. labels)* | 0.670 | 0.670 | 0.384 | — | — | — |

Forest full-polygon map, raw → guided-spatial (lifts macro-Dice, mostly via
`shallow_rock`): RF 0.728 → 0.740 (shallow 0.466 → 0.494); HGB 0.733 → 0.742
(shallow 0.466 → 0.486).

The dominant error for every model is `shallow_rock` ↔ `sand` confusion (see the
off-diagonals in `confusion_matrices.png`); `rock` recall is ≥ 0.89 throughout.
