"""LOPO cross-validation for the hierarchical U-Net.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/Scripts/python -m seabed_unet.crossval_hier \
      --config training/config/experiment_hier.yaml --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import load_config
from .crossval import FOLD_ORDER, fold_config, lopo_folds, print_table, summarize
from .evaluate_hier import evaluate_hier_checkpoint
from .logging_utils import add_file_handler, remove_handler, setup_logging
from .train_hier import run_hier_training

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def run_lopo_hier(cfg, limit=None) -> dict:
    polygons = cfg.split.polygons or FOLD_ORDER
    folds = lopo_folds(polygons)
    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]
    lopo_name = f"{cfg.name}_lopo"
    lopo_dir = cfg.base_dir / cfg.runs_dir / lopo_name
    lopo_dir.mkdir(parents=True, exist_ok=True)

    setup_logging()
    handler = add_file_handler(lopo_dir / "hier_crossval.log")
    try:
        fold_reports: dict[str, dict] = {}
        for fold in folds:
            test_poly = fold["test"][0]
            logger.info(f"\n[+] ===== fold test={test_poly} val={fold['val'][0]} "
                        f"train={fold['train']} =====")
            fcfg = fold_config(cfg, fold, lopo_name)
            run_hier_training(fcfg, limit=limit)
            report = evaluate_hier_checkpoint(fcfg, split="test")
            fold_reports[test_poly] = report

        summary = summarize(fold_reports, class_names)
        (lopo_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        logger.info("\n[+] LOPO summary — hierarchical U-Net")
        print_table(summary)
        logger.info(f"[+] summary -> {lopo_dir}")
        return summary
    finally:
        remove_handler(handler)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="LOPO cross-validation for hierarchical U-Net.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)
    if args.device:
        cfg.train.device = args.device
    run_lopo_hier(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
