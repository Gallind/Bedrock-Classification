"""Evaluation CLI for the hierarchical classifier.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/Scripts/python -m seabed_forest.evaluate_hier \
      --config training/config/forest_hier.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from seabed_unet.config import Config
from seabed_unet.data import load_split_records
from seabed_unet.logging_utils import add_file_handler, remove_handler, setup_logging
from seabed_unet.metrics import confusion_matrix, metrics_report
from seabed_unet.normalize import load_stats

from .config import ForestConfig, load_forest_config
from .hierarchical import load_hierarchical, predict
from .pixels import build_pixel_table

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def evaluate_hier_run(cfg: Config, forest: ForestConfig, split: str = "test") -> dict[str, dict]:
    setup_logging()
    run_dir = cfg.run_dir
    handler = add_file_handler(run_dir / "hier_eval.log")
    try:
        return _evaluate(cfg, forest, run_dir, split)
    finally:
        remove_handler(handler)


def _evaluate(cfg: Config, forest: ForestConfig, run_dir: Path, split: str) -> dict[str, dict]:
    band_modes = cfg.normalization.modes_for(cfg.bands)
    stats = load_stats(run_dir / "normalization_stats.json")
    splits = load_split_records(cfg)
    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]

    X, y, _ = build_pixel_table(
        splits[split], cfg.bands, cfg.class_ids, stats, cfg.feature_nodata,
        cfg.ignore_label, band_modes, dedup=False,
    )

    reports: dict[str, dict] = {}
    for kind in forest.models:
        est1, est2 = load_hierarchical(run_dir, kind)
        pred = predict(est1, est2, X, cfg.num_classes)
        cm = confusion_matrix(pred, y, cfg.num_classes)
        report = metrics_report(cm, class_names)
        (run_dir / f"metrics_{kind}.json").write_text(json.dumps(report, indent=2))
        reports[kind] = report
        logger.info(f"    {kind}: macroDice {report['macro_dice']:.4f} "
                    f"OA {report['overall_accuracy']:.4f} kappa {report['cohens_kappa']:.4f}")
        for cls in class_names:
            logger.info(f"      {cls}: dice {report['per_class'][cls]['dice']:.4f}")

    return reports


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the hierarchical seabed classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args(argv)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg, forest = load_forest_config(args.config, base_dir=base)
    evaluate_hier_run(cfg, forest, split=args.split)


if __name__ == "__main__":
    main()
