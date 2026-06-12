"""Live inference viewer: watch the model classify a polygon's tiles.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_unet.watch \
      --config training/config/experiment_3band.yaml --polygon polygon4 --delay 0.4

Opens a matplotlib window stepping through the polygon's base (_rot) tiles in
manifest (row/col) order: input bands | model prediction | ground truth on top,
and the blended classified map painting itself in below (same mask-weighted
softmax blending as seabed_unet.predict, re-argmaxed each step so overlapping
tiles refine live). Each step shows the tile's macro Dice against the labels.

--save also writes an animated GIF (half resolution) next to the run's maps.
Closing the window stops the session cleanly.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from seabed_tiler.viz import label_to_rgb

from .config import load_config
from .data import encode_target, load_run_records
from .inference import load_checkpoint, predict_probs
from .logging_utils import add_file_handler, setup_logging
from .metrics import confusion_matrix, dice_per_class
from .normalize import apply_stats, load_stats
from .predict import MapAccumulator, feature_valid_mask, resolve_polygon_stats
from .train import resolve_device

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def compose_input_rgb(inputs: np.ndarray) -> np.ndarray:
    """(B, H, W) normalized [0,1] features -> (H, W, 3) uint8 composite.

    First three bands map to R, G, B; missing bands (2-band experiments) are
    zero-filled so the composite stays interpretable rather than failing.
    """
    _, h, w = inputs.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(min(3, inputs.shape[0])):
        rgb[..., c] = inputs[c]
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


def masked_label_rgb(class_id_map: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Class colors with feature-invalid pixels blacked out."""
    rgb = label_to_rgb(class_id_map)
    rgb[~valid] = 0
    return rgb


def tile_macro_dice(
    pred_channels: np.ndarray, target: np.ndarray, num_classes: int
) -> float:
    """Macro Dice of one tile's prediction vs its encoded target (NaN if unlabeled)."""
    cm = confusion_matrix(pred_channels, target, num_classes)
    return float(np.nanmean(dice_per_class(cm)))


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Watch the seabed U-Net classify tiles live.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--checkpoint", default=None, help="Default: <run_dir>/best.pt")
    parser.add_argument("--polygon", default=None, help="Default: first test/split polygon.")
    parser.add_argument("--delay", type=float, default=0.4, help="Seconds per tile.")
    parser.add_argument("--save", action="store_true",
                        help="Also write <run_dir>/maps/<polygon>_watch.gif")
    args = parser.parse_args(argv)

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)
    setup_logging()
    add_file_handler(cfg.run_dir / "watch.log")

    fallback = cfg.split.test[0] if cfg.split.test else cfg.split.polygons[0]
    polygon = args.polygon or fallback
    device = resolve_device(cfg.train.device)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.run_dir / "best.pt"
    model, ckpt = load_checkpoint(ckpt_path, device)
    if ckpt["config"]["bands"] != cfg.bands:
        raise SystemExit(
            f"checkpoint was trained on bands {ckpt['config']['bands']}, "
            f"config asks for {cfg.bands}"
        )

    records = load_run_records(
        cfg.rot_dir(polygon), polygon, cfg.bands, cfg.base_dir, augmented=False
    )
    if not records:
        raise SystemExit(f"no tiles found for {polygon}")
    logger.info(f"[+] {cfg.name}: watching {polygon} — {len(records)} tiles, "
                f"checkpoint epoch {ckpt['epoch']} (val macro-Dice {ckpt['val_macro_dice']:.4f})")

    band_modes = cfg.normalization.modes_for(cfg.bands)
    stats = resolve_polygon_stats(
        cfg, polygon, records, load_stats(cfg.run_dir / "normalization_stats.json"), band_modes
    )
    acc = MapAccumulator(records, cfg.class_ids, cfg.feature_nodata)
    id_lookup = np.array(cfg.class_ids, dtype=np.uint8)

    import matplotlib.pyplot as plt

    plt.ion()
    fig = plt.figure(figsize=(11, 8))
    fig.canvas.manager.set_window_title(f"seabed_unet — {cfg.name} / {polygon}")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.5])
    panels = {}
    for i, title in enumerate(["inputs", "model prediction", "ground truth"]):
        ax = fig.add_subplot(gs[0, i])
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
        panels[title] = ax.imshow(np.zeros((2, 2, 3), dtype=np.uint8))
    ax_map = fig.add_subplot(gs[1, :])
    ax_map.set_title("classified map (blending in)", fontsize=10)
    ax_map.set_axis_off()
    map_artist = ax_map.imshow(np.zeros((acc.n_rows, acc.n_cols, 3), dtype=np.uint8))
    fig.tight_layout(rect=(0, 0, 1, 0.94))  # leave headroom for the suptitle
    plt.show(block=False)

    frames = []
    dices = []
    t0 = time.monotonic()
    for i, r in enumerate(records, start=1):
        inputs = apply_stats(r.features, r.polygon, cfg.bands, stats, cfg.feature_nodata, band_modes)
        probs = predict_probs(model, inputs, device)
        valid = feature_valid_mask(r.features, cfg.feature_nodata)
        pred_channels = probs.argmax(axis=0)
        pred_ids = id_lookup[pred_channels]
        target = encode_target(
            r.label, r.features, cfg.class_ids, cfg.feature_nodata, cfg.ignore_label
        )
        dice = tile_macro_dice(pred_channels, target, cfg.num_classes)
        if not np.isnan(dice):
            dices.append(dice)
        acc.add(r, probs)

        panels["inputs"].set_data(compose_input_rgb(inputs))
        panels["model prediction"].set_data(masked_label_rgb(pred_ids, valid))
        panels["ground truth"].set_data(label_to_rgb(r.label))
        map_artist.set_data(label_to_rgb(acc.class_map()))
        dice_txt = "n/a" if np.isnan(dice) else f"{dice:.2f}"
        fig.suptitle(
            f"tile {i}/{len(records)}  ·  {r.tile_id}  ·  tile Dice {dice_txt}", fontsize=11
        )
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(max(args.delay, 0.001))

        if args.save:
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            frames.append(buf[::2, ::2].copy())  # half resolution keeps GIFs sane

        logger.info(f"    tile {i}/{len(records)} {r.tile_id}  dice {dice_txt}")
        if not plt.fignum_exists(fig.number):
            logger.info("[+] window closed — stopping")
            break

    elapsed = time.monotonic() - t0
    mean_dice = float(np.mean(dices)) if dices else float("nan")
    logger.info(f"[+] watched {len(dices)} labeled tiles in {elapsed:.1f}s — "
                f"mean tile Dice {mean_dice:.3f}")

    if args.save and frames:
        from PIL import Image

        gif_path = cfg.run_dir / "maps" / f"{polygon}_watch.gif"
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(f) for f in frames]
        images[0].save(
            gif_path, save_all=True, append_images=images[1:],
            duration=int(max(args.delay, 0.15) * 1000), loop=0,
        )
        logger.info(f"[+] animation -> {gif_path}")

    if plt.fignum_exists(fig.number):
        fig.suptitle(fig._suptitle.get_text() + "   (done — close window to exit)")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
