"""Evaluation CLI: per-model test metrics + an RF/HGB/U-Net comparison table.

Metrics come from seabed_unet.metrics (identical to the U-Net's) over the same
test-split tile pixels. The U-Net reference row is INDICATIVE: its published numbers
are on four polygons (1/3/4/5); the forest here also trains on polygon6 (spec §7).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

from seabed_unet.config import Config
from seabed_unet.data import load_split_records
from seabed_unet.logging_utils import add_file_handler, remove_handler, setup_logging
from seabed_unet.metrics import confusion_matrix, metrics_report
from seabed_unet.normalize import load_stats

from .config import ForestConfig, load_forest_config
from .model import load_model
from .pixels import build_pixel_table

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)

# Published U-Net 3-band dev (spatial_blocks) numbers — indicative reference only.
UNET_3BAND_REFERENCE = {
    "macro_dice": 0.784, "overall_accuracy": 0.782, "cohens_kappa": 0.599,
}


def evaluate_run(cfg: Config, forest: ForestConfig, split: str = "test") -> dict[str, dict]:
    """Score every trained model on ``split``; write metrics_<kind>.json + comparison.{csv,md}."""
    setup_logging()
    run_dir = cfg.run_dir
    handler = add_file_handler(run_dir / "forest_eval.log")
    try:
        return _evaluate_run(cfg, forest, run_dir, split)
    finally:
        remove_handler(handler)


def _evaluate_run(cfg: Config, forest: ForestConfig, run_dir: Path, split: str) -> dict[str, dict]:
    band_modes = cfg.normalization.modes_for(cfg.bands)
    stats = load_stats(run_dir / "normalization_stats.json")
    splits = load_split_records(cfg)
    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]

    X, y, _ = build_pixel_table(
        splits[split], cfg.bands, cfg.class_ids, stats, cfg.feature_nodata,
        cfg.ignore_label, band_modes, dedup=False,   # eval over the raw test pixels
    )

    reports: dict[str, dict] = {}
    for kind in forest.models:
        est = load_model(run_dir / f"model_{kind}.joblib")
        pred = est.predict(X)
        cm = confusion_matrix(pred, y, cfg.num_classes)
        report = metrics_report(cm, class_names)
        (run_dir / f"metrics_{kind}.json").write_text(json.dumps(report, indent=2))
        reports[kind] = report
        logger.info(f"    {kind}: macroDice {report['macro_dice']:.4f} "
                    f"OA {report['overall_accuracy']:.4f} kappa {report['cohens_kappa']:.4f}")

    _write_comparison(run_dir, reports, class_names)
    logger.info(f"[+] metrics + comparison -> {run_dir}")
    return reports


def _write_comparison(run_dir: Path, reports: dict[str, dict], class_names: list[str]) -> None:
    header = ["model", "macro_dice", "overall_accuracy", "cohens_kappa"] + [f"dice_{n}" for n in class_names]

    def row(name: str, rep: dict) -> list[str]:
        cells = [name,
                 f"{rep['macro_dice']:.3f}",
                 f"{rep['overall_accuracy']:.3f}",
                 f"{rep['cohens_kappa']:.3f}"]
        for n in class_names:
            cells.append(f"{rep['per_class'][n]['dice']:.3f}" if "per_class" in rep else "—")
        return cells

    rows = [row(kind, reports[kind]) for kind in reports]
    rows.append(row("unet_3band_reference", UNET_3BAND_REFERENCE))  # per-class -> "—"

    with open(run_dir / "comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    md = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    md += ["| " + " | ".join(r) + " |" for r in rows]
    (run_dir / "comparison.md").write_text("\n".join(md) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the seabed per-pixel tree baseline.")
    parser.add_argument("--config", required=True, help="Forest experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args(argv)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg, forest = load_forest_config(args.config, base_dir=base)
    evaluate_run(cfg, forest, split=args.split)


if __name__ == "__main__":
    main()
