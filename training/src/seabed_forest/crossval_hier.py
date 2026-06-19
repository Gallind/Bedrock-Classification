"""LOPO cross-validation for the hierarchical classifier.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/Scripts/python -m seabed_forest.crossval_hier \
      --config training/config/forest_hier.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from seabed_unet.crossval import FOLD_ORDER, fold_config, lopo_folds, print_table, summarize
from seabed_unet.logging_utils import add_file_handler, remove_handler, setup_logging

from .config import load_forest_config
from .evaluate_hier import evaluate_hier_run
from .train_hier import train_hier_run

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def run_lopo_hier(cfg, forest, limit=None) -> dict:
    polygons = cfg.split.polygons or FOLD_ORDER
    folds = lopo_folds(polygons)
    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]
    lopo_name = f"{cfg.name}_lopo"
    lopo_dir = cfg.base_dir / cfg.runs_dir / lopo_name
    lopo_dir.mkdir(parents=True, exist_ok=True)

    setup_logging()
    handler = add_file_handler(lopo_dir / "hier_crossval.log")
    try:
        per_kind: dict[str, dict[str, dict]] = {kind: {} for kind in forest.models}

        for fold in folds:
            test_poly = fold["test"][0]
            logger.info(f"\n[+] ===== fold test={test_poly} val={fold['val'][0]} "
                        f"train={fold['train']} =====")
            fcfg = fold_config(cfg, fold, lopo_name)
            train_hier_run(fcfg, forest, limit=limit)
            reports = evaluate_hier_run(fcfg, forest, split="test")
            for kind, report in reports.items():
                per_kind[kind][test_poly] = report

        summaries: dict[str, dict] = {}
        for kind, fold_reports in per_kind.items():
            summary = summarize(fold_reports, class_names)
            (lopo_dir / f"summary_{kind}.json").write_text(json.dumps(summary, indent=2))
            summaries[kind] = summary
            logger.info(f"\n[+] LOPO summary — {kind}")
            print_table(summary)

        logger.info(f"\n[+] LOPO summaries -> {lopo_dir}")
        return summaries
    finally:
        remove_handler(handler)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="LOPO cross-validation for the hierarchical classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg, forest = load_forest_config(args.config, base_dir=base)
    run_lopo_hier(cfg, forest, limit=args.limit)


if __name__ == "__main__":
    main()
