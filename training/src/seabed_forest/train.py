"""Training CLI for the per-pixel tree baseline.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_forest.train \
      --config training/config/forest_3band.yaml

Writes to <runs_dir>/<name>/: model_<kind>.joblib (one per forest.models),
normalization_stats.json (for inference), feature_importance_<kind>.{csv,png}.
The run directory is wiped first (same policy as the U-Net trainer).
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

import numpy as np

from seabed_unet.config import Config
from seabed_unet.data import features_by_polygon, load_split_records
from seabed_unet.logging_utils import add_file_handler, remove_handler, setup_logging
from seabed_unet.normalize import compute_stats, save_stats

from .config import ForestConfig, load_forest_config
from .model import build_estimator, feature_importance, fit_estimator, save_model
from .pixels import build_pixel_table

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def _save_importance(importance: dict[str, float], run_dir: Path, kind: str) -> None:
    csv_path = run_dir / f"feature_importance_{kind}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["band", "importance"])
        for band, val in importance.items():
            w.writerow([band, round(val, 6)])
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(4, 3))
        bands = list(importance)
        ax.barh(bands, [importance[b] for b in bands], color="steelblue")
        ax.set_title(f"feature importance — {kind}")
        ax.set_xlabel("importance")
        fig.tight_layout()
        fig.savefig(run_dir / f"feature_importance_{kind}.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # plotting is a nicety, never fail training on it
        logger.warning(f"    importance plot skipped ({kind}): {exc}")


def train_run(cfg: Config, forest: ForestConfig, limit: int | None = None) -> dict:
    """Fit every model in forest.models from a resolved config; return a summary dict.
    Reused by the CLI and by seabed_forest.crossval (per-fold configs)."""
    setup_logging()
    run_dir = cfg.run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    handler = add_file_handler(run_dir / "forest_train.log")
    try:
        return _train_run(cfg, forest, run_dir, limit)
    finally:
        remove_handler(handler)


def _train_run(cfg: Config, forest: ForestConfig, run_dir: Path, limit: int | None) -> dict:
    band_modes = cfg.normalization.modes_for(cfg.bands)
    splits = load_split_records(cfg)
    if limit is not None:
        splits = {k: v[:limit] for k, v in splits.items()}

    # Forest trains on BASE (_rot) tiles only — augmentation is a no-op for a
    # context-free per-pixel model (spec §6.1). Enforce in code, not just YAML.
    train_records = [r for r in splits["train"] if not r.augmented]
    n_base = len(train_records)
    n_all = len(splits["train"])

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
    counts = {cfg.id_to_name[cfg.class_ids[c]]: int(n)
              for c, n in zip(*np.unique(y, return_counts=True))} if y.size else {}
    logger.info(f"[+] {cfg.name}: {len(train_records)} base tiles -> {X.shape[0]:,} train pixels "
                f"(dedup={forest.dedup_overlap}) class counts {counts}")

    for kind in forest.models:
        est = fit_estimator(build_estimator(kind, forest, cfg.num_classes), kind, X, y)
        save_model(est, run_dir / f"model_{kind}.joblib")
        importance = feature_importance(est, kind, X, y, cfg.bands, forest.seed)
        _save_importance(importance, run_dir, kind)
        logger.info(f"    {kind}: fitted; importance {importance}")

    logger.info(f"[+] models -> {run_dir}")
    return {
        "name": cfg.name, "models": list(forest.models),
        "n_train_pixels": int(X.shape[0]), "n_train_pixels_base_only": int(X.shape[0]),
        "n_base_tiles": n_base, "n_all_train_tiles": n_all, "run_dir": str(run_dir),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the seabed per-pixel tree baseline.")
    parser.add_argument("--config", required=True, help="Forest experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap tiles per split (smoke runs only).")
    args = parser.parse_args(argv)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg, forest = load_forest_config(args.config, base_dir=base)
    train_run(cfg, forest, limit=args.limit)


if __name__ == "__main__":
    main()
