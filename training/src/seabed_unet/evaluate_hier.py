"""Evaluation for the two-stage hierarchical U-Net.

Loads stage1/best.pt and stage2/best.pt from the run directory, runs inference
in two passes, combines into 3-class predictions, and scores against ground truth.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/Scripts/python -m seabed_unet.evaluate_hier \
      --config training/config/experiment_hier.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import load_split_records
from .evaluate import save_confusion_png
from .hier_dataset import ROCK_CH, SAND_CH, SHALLOW_CH, remap_stage1, remap_stage2
from .inference import load_checkpoint, predict_probs
from .logging_utils import add_file_handler, remove_handler, setup_logging
from .metrics import confusion_matrix, metrics_report
from .normalize import apply_stats, load_stats
from .data import encode_target
from .train import resolve_device

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)

IGNORE = -1


def predict_hier(model1, model2, x: np.ndarray, device: torch.device) -> np.ndarray:
    """(H, W) 3-class prediction combining both stages.

    Stage 1: binary rock/non-rock probabilities.
    Stage 2: binary shallow_rock/sand probabilities, applied only to non-rock pixels.
    Final label assigned per pixel by argmax of combined 3-class probability.
    """
    H, W = x.shape[1], x.shape[2]

    # Stage 1 probabilities: (2, H, W) -> [p_rock, p_nonrock]
    p1 = predict_probs(model1, x, device)   # (2, H, W)
    p_rock    = p1[0]                        # (H, W)
    p_nonrock = p1[1]                        # (H, W)

    # Stage 2 probabilities: (2, H, W) -> [p_shallow, p_sand]
    p2 = predict_probs(model2, x, device)   # (2, H, W)
    p_shallow = p_nonrock * p2[0]
    p_sand    = p_nonrock * p2[1]

    # Stack into (3, H, W) and take argmax
    combined = np.stack([p_rock, p_shallow, p_sand], axis=0)
    return combined.argmax(axis=0).astype(np.int64)


def evaluate_hier_checkpoint(cfg, split: str = "test") -> dict:
    setup_logging()
    handler = add_file_handler(cfg.run_dir / f"eval_{split}_hier.log")
    try:
        return _evaluate(cfg, split)
    finally:
        remove_handler(handler)


def _evaluate(cfg, split: str) -> dict:
    device = resolve_device(cfg.train.device)

    model1, _ = load_checkpoint(cfg.run_dir / "stage1" / "best.pt", device)
    model2, _ = load_checkpoint(cfg.run_dir / "stage2" / "best.pt", device)
    model1.eval()
    model2.eval()
    logger.info(f"[+] {cfg.name}: hierarchical eval on split '{split}'")

    stats = load_stats(cfg.run_dir / "normalization_stats.json")
    band_modes = cfg.normalization.modes_for(cfg.bands)
    records = load_split_records(cfg)[split]
    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]

    cm = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    with torch.no_grad():
        for r in records:
            x = apply_stats(r.features, r.polygon, cfg.bands, stats,
                            cfg.feature_nodata, band_modes)
            y = encode_target(r.label, r.features, cfg.class_ids,
                              cfg.feature_nodata, cfg.ignore_label)
            pred = predict_hier(model1, model2, x, device)
            valid = y != IGNORE
            cm += confusion_matrix(pred[valid], y[valid], cfg.num_classes)

    report = metrics_report(cm, class_names)
    report["split"] = split
    report["polygons"] = sorted({r.polygon for r in records})
    report["n_tiles"] = len(records)

    logger.info(f"    macro dice       {report['macro_dice']:.4f}")
    logger.info(f"    overall accuracy {report['overall_accuracy']:.4f}")
    logger.info(f"    cohen's kappa    {report['cohens_kappa']:.4f}")
    for name in class_names:
        c = report["per_class"][name]
        logger.info(f"      {name:<13} dice {c['dice']:.4f}  ({c['support_px']:,} px)")

    out_dir = cfg.run_dir / f"eval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(class_names), comments="")
    save_confusion_png(cm, class_names, out_dir / "confusion_matrix.png")
    logger.info(f"[+] report -> {out_dir}")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate the hierarchical seabed U-Net.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)
    if args.device:
        cfg.train.device = args.device
    evaluate_hier_checkpoint(cfg, split=args.split)


if __name__ == "__main__":
    main()
