"""Training CLI.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_unet.train \
      --config training/config/experiment_3band.yaml

Writes to <runs_dir>/<name>/: best.pt (checkpoint at best val macro-Dice),
normalization_stats.json (for inference), history.csv (per-epoch curves).
The run directory is wiped first — reruns never mix artifacts (same policy as
the tiler's clean_run_dir).
"""

from __future__ import annotations

import argparse
import csv
import logging
import random
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, load_config
from .logging_utils import add_file_handler, remove_handler, setup_logging
from .data import features_by_polygon, load_split_records
from .dataset import TileDataset
from .losses import MaskedSegmentationLoss, compute_class_weights
from .metrics import confusion_matrix, dice_per_class
from .normalize import compute_stats, save_stats
from .unet import UNet

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_datasets(
    cfg: Config, limit: int | None = None
) -> tuple[TileDataset, TileDataset, dict, dict]:
    """(train_ds, val_ds, stats, splits). ``limit`` caps tiles/split for smoke runs."""
    splits = load_split_records(cfg)
    if limit is not None:
        splits = {k: v[:limit] for k, v in splits.items()}
    band_modes = cfg.normalization.modes_for(cfg.bands)
    train_features = [r.features for r in splits["train"] if not r.augmented]
    stats = compute_stats(
        features_by_polygon(splits),
        train_features,
        cfg.bands,
        band_modes,
        cfg.feature_nodata,
        tuple(cfg.normalization.clip_percentiles),
    )
    train_ds = TileDataset(
        splits["train"], cfg.bands, cfg.class_ids, stats, cfg.feature_nodata,
        cfg.ignore_label, band_modes, augment=cfg.train.d4_augment, seed=cfg.train.seed,
    )
    val_ds = TileDataset(
        splits["val"], cfg.bands, cfg.class_ids, stats, cfg.feature_nodata,
        cfg.ignore_label, band_modes, augment=False,
    )
    return train_ds, val_ds, stats, splits


@torch.no_grad()
def evaluate_macro_dice(
    model: torch.nn.Module, loader: DataLoader, num_classes: int, device: torch.device
) -> float:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for x, y in loader:
        logits = model(x.to(device))
        pred = logits.argmax(dim=1).cpu().numpy()
        cm += confusion_matrix(pred, y.numpy(), num_classes)
    return float(np.nanmean(dice_per_class(cm)))


def run_training(cfg: Config, limit: int | None = None) -> dict:
    """Train one model from a fully-resolved config; return a summary dict.

    Reused by the CLI below and by seabed_unet.crossval (per-fold configs).
    Logs to stdout and <run_dir>/train.log.
    """
    setup_logging()
    run_dir = cfg.run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    file_handler = add_file_handler(run_dir / "train.log")
    try:
        return _run_training(cfg, run_dir, limit)
    finally:
        remove_handler(file_handler)


def _run_training(cfg: Config, run_dir: Path, limit: int | None) -> dict:
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    logger.info(f"[+] {cfg.name}: bands={cfg.bands} device={device.type}")

    train_ds, val_ds, stats, _ = build_datasets(cfg, limit=limit)
    band_modes = cfg.normalization.modes_for(cfg.bands)
    logger.info(f"    train={len(train_ds)} tiles (D4={'on' if cfg.train.d4_augment else 'off'}) "
                f"val={len(val_ds)} tiles  split={cfg.split.mode}  norm={band_modes}")

    save_stats(stats, run_dir / "normalization_stats.json")

    if cfg.loss.class_weights == "auto":
        weights_np = compute_class_weights(train_ds.targets, cfg.num_classes)
        class_weights = torch.tensor(weights_np, device=device)
        named = {cfg.id_to_name[cid]: round(float(w), 3)
                 for cid, w in zip(cfg.class_ids, weights_np)}
        logger.info(f"    class weights {named}")
    else:
        class_weights = None

    loader_gen = torch.Generator().manual_seed(cfg.train.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True, generator=loader_gen
    )
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    model = UNet(
        in_channels=len(cfg.bands), num_classes=cfg.num_classes,
        base_filters=cfg.model.base_filters, depth=cfg.model.depth,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"    UNet depth={cfg.model.depth} base={cfg.model.base_filters} "
          f"params={n_params:,}")

    criterion = MaskedSegmentationLoss(
        cfg.loss.ce_weight, cfg.loss.dice_weight, class_weights
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=cfg.train.scheduler_factor, patience=cfg.train.scheduler_patience,
    )

    best_dice, best_epoch = -1.0, -1
    history: list[dict] = []
    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        total_loss, n_batches = 0.0, 0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        val_dice = evaluate_macro_dice(model, val_loader, cfg.num_classes, device)
        scheduler.step(val_dice)
        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {"epoch": epoch, "train_loss": round(train_loss, 5),
             "val_macro_dice": round(val_dice, 5), "lr": lr}
        )

        marker = ""
        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg.model_dump(mode="json"),
                    "epoch": epoch,
                    "val_macro_dice": val_dice,
                },
                run_dir / "best.pt",
            )
            marker = "  *best*"
        logger.info(f"    epoch {epoch:3d}  loss {train_loss:.4f}  "
              f"val dice {val_dice:.4f}  lr {lr:.2e}{marker}")

        if epoch - best_epoch >= cfg.train.early_stop_patience:
            logger.info(f"[+] early stop at epoch {epoch} "
                  f"(no val improvement for {cfg.train.early_stop_patience})")
            break

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    logger.info(f"[+] best val macro-Dice {best_dice:.4f} (epoch {best_epoch}) -> {run_dir}")
    return {
        "name": cfg.name,
        "best_val_macro_dice": best_dice,
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "run_dir": str(run_dir),
    }


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the seabed U-Net.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap tiles per split (smoke runs only — not a valid experiment).",
    )
    args = parser.parse_args(argv)

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    run_training(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
