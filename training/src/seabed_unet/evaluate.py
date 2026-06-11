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
from pathlib import Path

import numpy as np

from .config import load_config
from .data import load_split_records
from .dataset import TileDataset
from .inference import load_checkpoint, predict_probs
from .metrics import confusion_matrix, metrics_report
from .normalize import load_stats
from .train import resolve_device


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


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained seabed U-Net.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--checkpoint", default=None, help="Default: <run_dir>/best.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args(argv)

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)
    device = resolve_device(cfg.train.device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.run_dir / "best.pt"
    model, ckpt = load_checkpoint(ckpt_path, device)
    if ckpt["config"]["bands"] != cfg.bands:
        raise SystemExit(
            f"checkpoint was trained on bands {ckpt['config']['bands']}, "
            f"config asks for {cfg.bands}"
        )
    print(f"[+] {cfg.name}: checkpoint epoch {ckpt['epoch']} "
          f"(val macro-Dice {ckpt['val_macro_dice']:.4f}) on split '{args.split}'")

    stats = load_stats(cfg.run_dir / "normalization_stats.json")
    records = load_split_records(cfg)[args.split]
    ds = TileDataset(
        records, cfg.bands, cfg.class_ids, stats, cfg.feature_nodata,
        cfg.ignore_label, augment=False,
    )

    cm = np.zeros((cfg.num_classes, cfg.num_classes), dtype=np.int64)
    for i in range(len(ds)):
        x, y = ds[i]
        probs = predict_probs(model, x.numpy(), device)
        cm += confusion_matrix(probs.argmax(axis=0), y.numpy(), cfg.num_classes)

    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]
    report = metrics_report(cm, class_names)
    report["split"] = args.split
    report["polygons"] = getattr(cfg.split, args.split)
    report["n_tiles"] = len(ds)

    print(f"    tiles {len(ds)}  scored px {cm.sum():,}")
    print(f"    overall accuracy {report['overall_accuracy']:.4f}")
    print(f"    cohen's kappa    {report['cohens_kappa']:.4f}")
    print(f"    macro dice       {report['macro_dice']:.4f}")
    for name in class_names:
        c = report["per_class"][name]
        print(f"      {name:<13} dice {c['dice']:.4f}  PAcc {c['producers_accuracy']:.4f}  "
              f"UAcc {c['users_accuracy']:.4f}  ({c['support_px']:,} px)")

    out_dir = cfg.run_dir / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(class_names), comments="")
    save_confusion_png(cm, class_names, out_dir / "confusion_matrix.png")
    print(f"[+] report -> {out_dir}")


if __name__ == "__main__":
    main()
