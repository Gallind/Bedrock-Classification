"""Two-stage hierarchical U-Net training.

Stage 1: binary U-Net — rock vs non-rock (2 output classes).
Stage 2: binary U-Net — shallow_rock vs sand (2 output classes), rock pixels masked.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/Scripts/python -m seabed_unet.train_hier \
      --config training/config/experiment_hier.yaml --device cuda
"""

from __future__ import annotations

import argparse
import csv
import logging
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, load_config
from .data import features_by_polygon, load_split_records
from .hier_dataset import Stage1Dataset, Stage2Dataset
from .logging_utils import add_file_handler, remove_handler, setup_logging
from .losses import MaskedSegmentationLoss, compute_class_weights
from .metrics import confusion_matrix, dice_per_class
from .normalize import compute_stats, save_stats
from .train import resolve_device, seed_everything
from .unet import UNet

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def _train_stage(
    stage: int,
    train_ds,
    val_ds,
    cfg: Config,
    run_dir: Path,
    device: torch.device,
    num_classes: int = 2,
    class_names: list[str] | None = None,
) -> dict:
    """Generic training loop for one stage. Saves best.pt into run_dir/stage{stage}/."""
    stage_dir = run_dir / f"stage{stage}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    if cfg.loss.class_weights == "auto":
        weights_np = compute_class_weights(train_ds.targets, num_classes)
        class_weights = torch.tensor(weights_np, device=device)
        named = {n: round(float(w), 3) for n, w in zip(class_names or [], weights_np)}
        logger.info(f"    stage{stage} class weights {named}")
    else:
        class_weights = None

    loader_gen = torch.Generator().manual_seed(cfg.train.seed)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size,
                              shuffle=True, generator=loader_gen)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    model = UNet(
        in_channels=len(cfg.bands), num_classes=num_classes,
        base_filters=cfg.model.base_filters, depth=cfg.model.depth,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"    stage{stage} UNet: depth={cfg.model.depth} base={cfg.model.base_filters} "
                f"params={n_params:,}  train={len(train_ds)} val={len(val_ds)} tiles")

    criterion = MaskedSegmentationLoss(cfg.loss.ce_weight, cfg.loss.dice_weight, class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
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

        model.eval()
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        with torch.no_grad():
            for x, y in val_loader:
                pred = model(x.to(device)).argmax(dim=1).cpu().numpy()
                cm += confusion_matrix(pred, y.numpy(), num_classes)
        val_dice = float(np.nanmean(dice_per_class(cm)))

        scheduler.step(val_dice)
        lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_macro_dice": round(val_dice, 5), "lr": lr})

        marker = ""
        if val_dice > best_dice:
            best_dice, best_epoch = val_dice, epoch
            torch.save({"model_state": model.state_dict(),
                        "config": cfg.model_dump(mode="json"),
                        "epoch": epoch, "val_macro_dice": val_dice,
                        "stage": stage, "num_classes": num_classes},
                       stage_dir / "best.pt")
            marker = "  *best*"
        logger.info(f"    stage{stage} epoch {epoch:3d}  loss {train_loss:.4f}  "
                    f"val dice {val_dice:.4f}  lr {lr:.2e}{marker}")

        if epoch - best_epoch >= cfg.train.early_stop_patience:
            logger.info(f"    stage{stage} early stop at epoch {epoch}")
            break

    with open(stage_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    logger.info(f"    stage{stage} best val dice {best_dice:.4f} (epoch {best_epoch}) -> {stage_dir}")
    return {"best_val_macro_dice": best_dice, "best_epoch": best_epoch,
            "epochs_run": len(history), "stage_dir": str(stage_dir)}


def run_hier_training(cfg: Config, limit: int | None = None) -> dict:
    setup_logging()
    run_dir = cfg.run_dir
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    handler = add_file_handler(run_dir / "hier_train.log")
    try:
        return _run(cfg, run_dir, limit)
    finally:
        remove_handler(handler)


def _run(cfg: Config, run_dir: Path, limit: int | None) -> dict:
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    logger.info(f"[+] hierarchical U-Net: {cfg.name}  bands={cfg.bands}  device={device.type}")

    splits = load_split_records(cfg)
    if limit is not None:
        splits = {k: v[:limit] for k, v in splits.items()}

    band_modes = cfg.normalization.modes_for(cfg.bands)
    train_features = [r.features for r in splits["train"] if not r.augmented]
    stats = compute_stats(
        features_by_polygon(splits), train_features, cfg.bands, band_modes,
        cfg.feature_nodata, tuple(cfg.normalization.clip_percentiles),
    )
    save_stats(stats, run_dir / "normalization_stats.json")

    ds_kwargs = dict(bands=cfg.bands, class_ids=cfg.class_ids, stats=stats,
                     nodata=cfg.feature_nodata, ignore_label=cfg.ignore_label,
                     band_modes=band_modes, seed=cfg.train.seed)

    # Stage 1: rock vs non-rock
    logger.info("[+] Stage 1: rock vs non-rock")
    s1_train = Stage1Dataset(splits["train"], augment=cfg.train.d4_augment, **ds_kwargs)
    s1_val   = Stage1Dataset(splits["val"],   augment=False, **ds_kwargs)
    s1_result = _train_stage(1, s1_train, s1_val, cfg, run_dir, device,
                             num_classes=2, class_names=["rock", "non-rock"])

    # Stage 2: shallow_rock vs sand (non-rock tiles only)
    logger.info("[+] Stage 2: shallow_rock vs sand")
    s2_train = Stage2Dataset(splits["train"], augment=cfg.train.d4_augment, **ds_kwargs)
    s2_val   = Stage2Dataset(splits["val"],   augment=False, **ds_kwargs)
    logger.info(f"    stage2 train tiles: {len(s2_train)} (skipped {s2_train._skipped} rock-only)")
    logger.info(f"    stage2 val   tiles: {len(s2_val)}   (skipped {s2_val._skipped} rock-only)")
    s2_result = _train_stage(2, s2_train, s2_val, cfg, run_dir, device,
                             num_classes=2, class_names=["shallow_rock", "sand"])

    logger.info(f"[+] hierarchical training done -> {run_dir}")
    return {"name": cfg.name, "stage1": s1_result, "stage2": s2_result}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Train the hierarchical seabed U-Net.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)
    if args.device:
        cfg.train.device = args.device
    run_hier_training(cfg, limit=args.limit)


if __name__ == "__main__":
    main()
