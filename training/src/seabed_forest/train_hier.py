"""Training CLI for the two-stage hierarchical classifier.

Stage 1: rock vs non-rock.
Stage 2: sand vs shallow_rock (trained only on non-rock pixels).

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/Scripts/python -m seabed_forest.train_hier \
      --config training/config/forest_hier.yaml
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

import numpy as np

from seabed_unet.config import Config
from seabed_unet.data import features_by_polygon, load_split_records
from seabed_unet.logging_utils import add_file_handler, remove_handler, setup_logging
from seabed_unet.normalize import compute_stats, save_stats

from .config import ForestConfig, load_forest_config
from .hierarchical import (
    ROCK_CH, save_hierarchical, to_stage1, to_stage2,
)
from .model import build_estimator, feature_importance, fit_estimator
from .pixels import build_pixel_table
from .train import _save_importance

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def train_hier_run(cfg: Config, forest: ForestConfig, limit: int | None = None) -> dict:
    setup_logging()
    run_dir = cfg.run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    handler = add_file_handler(run_dir / "hier_train.log")
    try:
        return _train(cfg, forest, run_dir, limit)
    finally:
        remove_handler(handler)


def _train(cfg: Config, forest: ForestConfig, run_dir: Path, limit: int | None) -> dict:
    band_modes = cfg.normalization.modes_for(cfg.bands)
    splits = load_split_records(cfg)
    if limit is not None:
        splits = {k: v[:limit] for k, v in splits.items()}

    train_records = [r for r in splits["train"] if not r.augmented]

    train_features = [r.features for r in train_records]
    stats = compute_stats(
        features_by_polygon(splits), train_features, cfg.bands, band_modes,
        cfg.feature_nodata, tuple(cfg.normalization.clip_percentiles),
    )
    save_stats(stats, run_dir / "normalization_stats.json")

    X, y, _ = build_pixel_table(
        train_records, cfg.bands, cfg.class_ids, stats, cfg.feature_nodata,
        cfg.ignore_label, band_modes, dedup=forest.dedup_overlap,
        max_pixels_per_class=forest.max_pixels_per_class, seed=forest.seed,
    )

    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]
    counts = {class_names[c]: int(n) for c, n in zip(*np.unique(y, return_counts=True))} if y.size else {}
    logger.info(f"[+] {cfg.name}: {len(train_records)} base tiles -> {X.shape[0]:,} train pixels "
                f"class counts {counts}")

    # Stage 1 data: all pixels, binary rock/non-rock
    y1 = to_stage1(y)

    # Stage 2 data: non-rock pixels only, binary shallow_rock/sand
    nonrock_mask, y2 = to_stage2(y)
    X2 = X[nonrock_mask]

    s2_counts = {n: int(c) for n, c in zip(["shallow_rock", "sand"], np.bincount(y2, minlength=2))}
    logger.info(f"    stage1: {y1.shape[0]:,} pixels (rock={int((y1==0).sum()):,} non-rock={int((y1==1).sum()):,})")
    logger.info(f"    stage2: {y2.shape[0]:,} non-rock pixels {s2_counts}")

    for kind in forest.models:
        logger.info(f"  [{kind}] fitting stage 1 (rock vs non-rock) ...")
        est1 = fit_estimator(build_estimator(kind, forest, 2), kind, X, y1)

        logger.info(f"  [{kind}] fitting stage 2 (sand vs shallow_rock) ...")
        est2 = fit_estimator(build_estimator(kind, forest, 2), kind, X2, y2)

        save_hierarchical(est1, est2, run_dir, kind)

        imp1 = feature_importance(est1, kind, X, y1, cfg.bands, forest.seed)
        imp2 = feature_importance(est2, kind, X2, y2, cfg.bands, forest.seed)
        _save_importance(imp1, run_dir, f"{kind}_stage1")
        _save_importance(imp2, run_dir, f"{kind}_stage2")
        logger.info(f"    stage1 importance {imp1}")
        logger.info(f"    stage2 importance {imp2}")

    logger.info(f"[+] models -> {run_dir}")
    return {"name": cfg.name, "models": list(forest.models), "n_train_pixels": int(X.shape[0])}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the hierarchical seabed classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg, forest = load_forest_config(args.config, base_dir=base)
    train_hier_run(cfg, forest, limit=args.limit)


if __name__ == "__main__":
    main()
