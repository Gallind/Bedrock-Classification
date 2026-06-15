"""Leave-one-polygon-out cross-validation for the tree baseline (the honest
cross-survey number). Reuses the U-Net's fold definitions + summary so the table
format matches. Fold i: test = polygon_i, val = next (rotation), train = rest.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_forest.crossval \
      --config training/config/forest_3band.yaml

Outputs: <runs_dir>/<name>_lopo/fold_<polygon>/ (per fold) and
<runs_dir>/<name>_lopo/summary_<kind>.json + a printed table per model.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from seabed_unet.config import Config
from seabed_unet.crossval import FOLD_ORDER, fold_config, lopo_folds, print_table, summarize
from seabed_unet.logging_utils import add_file_handler, remove_handler, setup_logging

from .config import ForestConfig, load_forest_config
from .eval_spatial import evaluate_spatial
from .evaluate import evaluate_run
from .train import train_run

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def run_lopo(cfg: Config, forest: ForestConfig, limit: int | None = None) -> dict[str, dict]:
    """Train+evaluate every fold for every model; write per-model summaries. Returns
    {kind: summary_dict}."""
    polygons = cfg.split.polygons or FOLD_ORDER
    folds = lopo_folds(polygons)
    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]
    lopo_name = f"{cfg.name}_lopo"
    lopo_dir = cfg.base_dir / cfg.runs_dir / lopo_name
    lopo_dir.mkdir(parents=True, exist_ok=True)

    setup_logging()
    handler = add_file_handler(lopo_dir / "forest_crossval.log")
    try:
        # {kind: {test_polygon: metrics_report}}
        per_kind: dict[str, dict[str, dict]] = {kind: {} for kind in forest.models}
        spatial_enabled = forest.spatial.enabled
        if spatial_enabled:
            per_kind_raw: dict[str, dict[str, dict]] = {kind: {} for kind in forest.models}
            per_kind_spatial: dict[str, dict[str, dict]] = {kind: {} for kind in forest.models}
        for fold in folds:
            test_poly = fold["test"][0]
            logger.info(f"\n[+] ===== fold test={test_poly} val={fold['val'][0]} "
                        f"train={fold['train']} =====")
            fcfg = fold_config(cfg, fold, lopo_name)   # polygon-mode split; nested run dir
            train_run(fcfg, forest, limit=limit)
            reports = evaluate_run(fcfg, forest, split="test")
            for kind, report in reports.items():
                per_kind[kind][test_poly] = report
            if spatial_enabled:
                spatial_reports = evaluate_spatial(fcfg, forest, split="test")
                for kind in forest.models:
                    per_kind_raw[kind][test_poly] = spatial_reports[f"{kind}_map_raw"]
                    per_kind_spatial[kind][test_poly] = spatial_reports[f"{kind}_map_spatial"]

        summaries: dict[str, dict] = {}
        for kind, fold_reports in per_kind.items():
            summary = summarize(fold_reports, class_names)
            (lopo_dir / f"summary_{kind}.json").write_text(json.dumps(summary, indent=2))
            summaries[kind] = summary
            logger.info(f"\n[+] LOPO summary — {kind}")
            print_table(summary)
        if spatial_enabled:
            for kind in forest.models:
                raw_summary = summarize(per_kind_raw[kind], class_names)
                spatial_summary = summarize(per_kind_spatial[kind], class_names)
                combined = {"map_raw": raw_summary, "map_spatial": spatial_summary}
                (lopo_dir / f"spatial_summary_{kind}.json").write_text(json.dumps(combined, indent=2))
                logger.info(f"\n[+] LOPO spatial summary — {kind} "
                            f"(raw mDice {raw_summary['macro_dice']['mean']:.3f} -> "
                            f"spatial {spatial_summary['macro_dice']['mean']:.3f})")
        logger.info(f"\n[+] LOPO summaries -> {lopo_dir}")
        return summaries
    finally:
        remove_handler(handler)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="LOPO cross-validation for the tree baseline.")
    parser.add_argument("--config", required=True, help="Forest experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--limit", type=int, default=None, help="Cap tiles per split (smoke only).")
    args = parser.parse_args(argv)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg, forest = load_forest_config(args.config, base_dir=base)
    run_lopo(cfg, forest, limit=args.limit)


if __name__ == "__main__":
    main()
