"""Map-level spatial evaluation: raw-argmax map vs guided-regularized map.

For each model and each polygon present in ``split``, assemble that polygon's FULL
posterior over all its base (_rot) tiles (features only — no labels — so using
neighbouring tiles for context is not a leak; mirrors the U-Net predict blend), then
build a raw-argmax class map and a spatially-regularized class map. Score BOTH against
ground truth at the split's test-tile pixels (sampling each North-up map back to the
tile grid), so the spatial delta is measured on identical pixels using the same metrics
as the per-pixel evaluator.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from rasterio.warp import Resampling, reproject

from seabed_unet.config import Config
from seabed_unet.data import IGNORE_INDEX, encode_target, load_run_records, load_split_records
from seabed_unet.logging_utils import add_file_handler, remove_handler, setup_logging
from seabed_unet.metrics import confusion_matrix, metrics_report
from seabed_unet.normalize import apply_stats, load_stats
from seabed_unet.predict import feature_valid_mask, resolve_polygon_stats

from .config import ForestConfig, load_forest_config
from .model import load_model
from .predict import PosteriorMapAccumulator, _proba_field, class_map_from_prob
from .spatial import regularize_posterior

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def _channel_lookup(class_ids: list[int]) -> np.ndarray:
    """uint8 class-id -> channel index, with -1 for any id not in class_ids (incl. 0)."""
    lut = np.full(int(max(class_ids)) + 1, -1, dtype=np.int64)
    for ch, cid in enumerate(class_ids):
        lut[cid] = ch
    return lut


def _score_map_against_tile(class_map_id, map_transform, map_crs, tile, cfg, lut, num_classes):
    """Confusion matrix of one North-up class-id map sampled back onto one tile's pixels."""
    h, w = tile.label.shape
    pred_id = np.zeros((h, w), dtype=np.uint8)
    reproject(
        source=class_map_id, destination=pred_id,
        src_transform=map_transform, src_crs=map_crs,
        dst_transform=tile.transform, dst_crs=tile.crs,
        resampling=Resampling.nearest,
    )
    target = encode_target(tile.label, tile.features, cfg.class_ids, cfg.feature_nodata, cfg.ignore_label)
    pred_channel = lut[pred_id]                      # -1 where map uncovered (id 0)
    # only score where the tile is trainable AND the map made a prediction
    target = np.where(pred_channel >= 0, target, IGNORE_INDEX)
    return confusion_matrix(np.clip(pred_channel, 0, None), target, num_classes)


def evaluate_spatial(cfg: Config, forest: ForestConfig, split: str = "test") -> dict[str, dict]:
    """Score raw vs spatially-regularized maps per model on ``split`` test pixels."""
    setup_logging()
    run_dir = cfg.run_dir
    handler = add_file_handler(run_dir / "forest_eval_spatial.log")
    try:
        return _evaluate_spatial(cfg, forest, run_dir, split)
    finally:
        remove_handler(handler)


def _evaluate_spatial(cfg: Config, forest: ForestConfig, run_dir: Path, split: str) -> dict[str, dict]:
    if forest.spatial.guide_band not in cfg.bands:
        raise ValueError(f"spatial.guide_band {forest.spatial.guide_band!r} not in bands {cfg.bands}")
    guide_idx = cfg.bands.index(forest.spatial.guide_band)
    band_modes = cfg.normalization.modes_for(cfg.bands)
    base_stats = load_stats(run_dir / "normalization_stats.json")
    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]
    lut = _channel_lookup(cfg.class_ids)

    splits = load_split_records(cfg)
    test_by_poly: dict[str, list] = {}
    for r in splits[split]:
        test_by_poly.setdefault(r.polygon, []).append(r)

    reports: dict[str, dict] = {}
    for kind in forest.models:
        est = load_model(run_dir / f"model_{kind}.joblib")
        cm_raw = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
        cm_spatial = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
        for poly, test_tiles in test_by_poly.items():
            recs = load_run_records(cfg.rot_dir(poly), poly, cfg.bands, cfg.base_dir, augmented=False)
            stats = resolve_polygon_stats(cfg, poly, recs, base_stats, band_modes)
            acc = PosteriorMapAccumulator(recs, cfg.class_ids, cfg.feature_nodata)
            for r in recs:
                inputs = apply_stats(r.features, r.polygon, cfg.bands, stats, cfg.feature_nodata, band_modes)
                acc.add(r, _proba_field(est, inputs, cfg.num_classes), inputs[guide_idx])
            prob, guide, covered = acc.posterior()
            raw_map = class_map_from_prob(prob, covered, cfg.class_ids)
            prob_reg = regularize_posterior(prob, guide, covered, forest.spatial.radius, forest.spatial.eps)
            spatial_map = class_map_from_prob(prob_reg, covered, cfg.class_ids)
            for tile in test_tiles:
                cm_raw += _score_map_against_tile(raw_map, acc.transform, acc.crs, tile, cfg, lut, cfg.num_classes)
                cm_spatial += _score_map_against_tile(spatial_map, acc.transform, acc.crs, tile, cfg, lut, cfg.num_classes)
        rep_raw = metrics_report(cm_raw, class_names)
        rep_spatial = metrics_report(cm_spatial, class_names)
        (run_dir / f"metrics_{kind}_map_raw.json").write_text(json.dumps(rep_raw, indent=2))
        (run_dir / f"metrics_{kind}_map_spatial.json").write_text(json.dumps(rep_spatial, indent=2))
        reports[f"{kind}_map_raw"] = rep_raw
        reports[f"{kind}_map_spatial"] = rep_spatial
        logger.info(f"    {kind}: raw mDice {rep_raw['macro_dice']:.4f} -> "
                    f"spatial mDice {rep_spatial['macro_dice']:.4f}")

    _write_spatial_comparison(run_dir, reports, class_names)
    logger.info(f"[+] spatial comparison -> {run_dir}")
    return reports


def _write_spatial_comparison(run_dir: Path, reports: dict[str, dict], class_names: list[str]) -> None:
    header = ["variant", "macro_dice", "overall_accuracy", "cohens_kappa"] + [f"dice_{n}" for n in class_names]
    rows = []
    for variant, rep in reports.items():
        cells = [variant, f"{rep['macro_dice']:.3f}", f"{rep['overall_accuracy']:.3f}",
                 f"{rep['cohens_kappa']:.3f}"] + [f"{rep['per_class'][n]['dice']:.3f}" for n in class_names]
        rows.append(cells)
    with open(run_dir / "spatial_comparison.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (run_dir / "spatial_comparison.md").write_text("\n".join(md) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Map-level raw-vs-spatial evaluation for the tree baseline.")
    parser.add_argument("--config", required=True, help="Forest experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args(argv)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg, forest = load_forest_config(args.config, base_dir=base)
    evaluate_spatial(cfg, forest, split=args.split)


if __name__ == "__main__":
    main()
