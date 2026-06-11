"""Evaluation CLI: metrics report + confusion matrix for a trained checkpoint.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_unet.evaluate \
      --config training/config/experiment_3band.yaml

Evaluates the test split's base (_rot) tiles — never augmented ones — and writes
metrics.json + confusion_matrix.{csv,png} to <run_dir>/eval_<split>/.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from .config import load_config
from .data import load_split_records
from .dataset import TileDataset
from .inference import load_checkpoint, predict_probs
from .logging_utils import add_file_handler, remove_handler, setup_logging
from .metrics import confusion_matrix, metrics_report
from .normalize import load_stats
from .train import resolve_device

logger = logging.getLogger(__name__)


def save_confusion_png(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("reference")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def evaluate_checkpoint(
    cfg, split: str = "test", checkpoint: str | Path | None = None
) -> dict:
    """Score a checkpoint on a split's base tiles; write artifacts; return the report.

    Reused by the CLI below and by seabed_unet.crossval (per-fold configs).
    Logs to stdout and <run_dir>/eval_<split>.log.
    """
    setup_logging()
    file_handler = add_file_handler(cfg.run_dir / f"eval_{split}.log")
    try:
        return _evaluate_checkpoint(cfg, split, checkpoint)
    finally:
        remove_handler(file_handler)


def _evaluate_checkpoint(cfg, split: str, checkpoint: str | Path | None) -> dict:
    device = resolve_device(cfg.train.device)

    ckpt_path = Path(checkpoint) if checkpoint else cfg.run_dir / "best.pt"
    model, ckpt = load_checkpoint(ckpt_path, device)
    if ckpt["config"]["bands"] != cfg.bands:
        raise SystemExit(
            f"checkpoint was trained on bands {ckpt['config']['bands']}, "
            f"config asks for {cfg.bands}"
        )
    logger.info(f"[+] {cfg.name}: checkpoint epoch {ckpt['epoch']} "
          f"(val macro-Dice {ckpt['val_macro_dice']:.4f}) on split '{split}'")

    stats = load_stats(cfg.run_dir / "normalization_stats.json")
    records = load_split_records(cfg)[split]
    ds = TileDataset(
        records, cfg.bands, cfg.class_ids, stats, cfg.feature_nodata,
        cfg.ignore_label, cfg.normalization.modes_for(cfg.bands), augment=False,
    )

    cm = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    for i in range(len(ds)):
        x, y = ds[i]
        probs = predict_probs(model, x.numpy(), device)
        cm += confusion_matrix(probs.argmax(axis=0), y.numpy(), cfg.num_classes)

    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]
    report = metrics_report(cm, class_names)
    report["split"] = split
    report["split_mode"] = cfg.split.mode
    report["polygons"] = sorted({r.polygon for r in records})
    report["n_tiles"] = len(ds)

    logger.info(f"    tiles {len(ds)}  scored px {cm.sum():,}")
    logger.info(f"    overall accuracy {report['overall_accuracy']:.4f}")
    logger.info(f"    cohen's kappa    {report['cohens_kappa']:.4f}")
    logger.info(f"    macro dice       {report['macro_dice']:.4f}")
    for name in class_names:
        c = report["per_class"][name]
        logger.info(f"      {name:<13} dice {c['dice']:.4f}  PAcc {c['producers_accuracy']:.4f}  "
              f"UAcc {c['users_accuracy']:.4f}  ({c['support_px']:,} px)")

    out_dir = cfg.run_dir / f"eval_{split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(class_names), comments="")
    save_confusion_png(cm, class_names, out_dir / "confusion_matrix.png")
    logger.info(f"[+] report -> {out_dir}")
    return report


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained seabed U-Net.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--checkpoint", default=None, help="Default: <run_dir>/best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args(argv)

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)
    evaluate_checkpoint(cfg, split=args.split, checkpoint=args.checkpoint)


if __name__ == "__main__":
    main()
