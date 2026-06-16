"""Live inference viewer: watch the full polygon being classified tile by tile.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_unet.watch \
      --config training/config/experiment_3band.yaml --polygon polygon4 --delay 0.4

Top row: the CURRENT tile in each input band (backscatter grayscale, bathymetry
'summer', slope 'YlOrRd' — matching the project's render conventions) plus the
model's classification of that tile. Main panel: the full polygon rendered as a
grayscale survey backdrop, with class colors painting in as tiles are
classified (same mask-weighted softmax blending as seabed_unet.predict,
re-argmaxed each step so overlapping tiles refine live) and a yellow outline
marking the tile currently under the model.

--save also writes an animated GIF (half resolution) next to the run's maps.
Closing the window stops the session cleanly.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
from rasterio.enums import Resampling
from rasterio.warp import reproject

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

# Same per-band looks as the tiler's JPEG renders (viz.BAND_STYLE), minus hillshade.
BAND_CMAPS = {"backscatter": "gray", "bathymetry": "summer", "slope": "YlOrRd"}
OVERLAY_ALPHA = 0.8  # how strongly class colors cover the survey backdrop


def render_band(norm_band: np.ndarray, valid: np.ndarray, band_name: str) -> np.ndarray:
    """One normalized [0,1] band -> (H, W, 3) uint8 via its conventional colormap."""
    from matplotlib import colormaps

    cmap = colormaps[BAND_CMAPS.get(band_name, "gray")]
    rgb = (cmap(np.clip(norm_band, 0.0, 1.0))[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb


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


def build_backdrop(
    records, acc: MapAccumulator, bands, stats, band_modes, nodata: float
) -> np.ndarray:
    """(H, W, 3) uint8 grayscale mosaic of the whole polygon on the map grid.

    Mean of the normalized input bands, reprojected tile by tile with the same
    mask-weighted averaging as the class map — so backdrop and classification
    are guaranteed pixel-aligned. Uncovered pixels stay black.
    """
    lum_sum = np.zeros((acc.n_rows, acc.n_cols), dtype=np.float64)
    weight = np.zeros((acc.n_rows, acc.n_cols), dtype=np.float64)
    for r in records:
        inputs = apply_stats(r.features, r.polygon, bands, stats, nodata, band_modes)
        valid = feature_valid_mask(r.features, nodata).astype(np.float32)
        lum = inputs.mean(axis=0) * valid
        dst = np.zeros((2, acc.n_rows, acc.n_cols), dtype=np.float32)
        for b, src in enumerate((lum, valid)):
            reproject(
                source=src, destination=dst[b],
                src_transform=r.transform, src_crs=r.crs,
                dst_transform=acc.transform, dst_crs=acc.crs,
                resampling=Resampling.bilinear,
            )
        lum_sum += dst[0]
        weight += dst[1]
    lum = np.where(weight > 1e-3, lum_sum / np.maximum(weight, 1e-9), 0.0)
    gray = (np.clip(lum, 0.0, 1.0) * 255).astype(np.uint8)
    return np.repeat(gray[..., np.newaxis], 3, axis=2)


def build_truth_map(records, acc: MapAccumulator) -> np.ndarray:
    """(H, W) uint8 ground-truth class map mosaicked onto the same grid.

    Nearest resampling only — class ids must never interpolate. Overlapping
    tiles carry identical labels (cut from one master grid), so last-wins is
    safe; uncovered pixels stay 0 (unlabeled).
    """
    truth = np.zeros((acc.n_rows, acc.n_cols), dtype=np.uint8)
    for r in records:
        dst = np.zeros((acc.n_rows, acc.n_cols), dtype=np.float32)
        reproject(
            source=r.label.astype(np.float32), destination=dst,
            src_transform=r.transform, src_crs=r.crs,
            dst_transform=acc.transform, dst_crs=acc.crs,
            resampling=Resampling.nearest,
        )
        labeled = dst > 0
        truth[labeled] = dst[labeled].astype(np.uint8)
    return truth


def class_overlay(
    backdrop: np.ndarray, class_map: np.ndarray, alpha: float = OVERLAY_ALPHA
) -> np.ndarray:
    """Paint classified pixels over the backdrop; unclassified pixels keep terrain."""
    out = backdrop.copy()
    classified = class_map > 0
    colors = label_to_rgb(class_map)
    out[classified] = (
        alpha * colors[classified].astype(np.float32)
        + (1.0 - alpha) * backdrop[classified].astype(np.float32)
    ).astype(np.uint8)
    return out


def tile_outline_px(record, acc: MapAccumulator) -> np.ndarray:
    """(4, 2) tile corner positions in map-pixel (col, row) coords for the marker."""
    h, w = record.label.shape
    res = acc.transform.a
    xmin, ymax = acc.transform.c, acc.transform.f
    corners = [record.transform * c for c in [(0, 0), (w, 0), (w, h), (0, h)]]
    return np.array([[(x - xmin) / res, (ymax - y) / res] for x, y in corners])


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

    logger.info(f"    building survey backdrop + truth map ({len(records)} tiles) ...")
    backdrop = build_backdrop(records, acc, cfg.bands, stats, band_modes, cfg.feature_nodata)
    truth_rgb = class_overlay(backdrop, build_truth_map(records, acc))

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    plt.ion()
    fig = plt.figure(figsize=(14, 9))
    fig.canvas.manager.set_window_title(f"seabed_unet — {cfg.name} / {polygon}")
    n_top = len(cfg.bands) + 1
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.2])
    gs_top = outer[0].subgridspec(1, n_top)
    gs_bottom = outer[1].subgridspec(1, 2)
    panels = {}
    for i, band in enumerate(cfg.bands):
        ax = fig.add_subplot(gs_top[0, i])
        ax.set_title(f"tile · {band}", fontsize=10)
        ax.set_axis_off()
        panels[band] = ax.imshow(np.zeros((2, 2, 3), dtype=np.uint8))
    ax = fig.add_subplot(gs_top[0, n_top - 1])
    ax.set_title("tile · classified", fontsize=10)
    ax.set_axis_off()
    panels["pred"] = ax.imshow(np.zeros((2, 2, 3), dtype=np.uint8))

    ax_map = fig.add_subplot(gs_bottom[0, 0])
    ax_map.set_title(f"{polygon} — model (filling in)", fontsize=10)
    ax_map.set_axis_off()
    map_artist = ax_map.imshow(backdrop)
    outline = MplPolygon(
        np.zeros((4, 2)), closed=True, fill=False, edgecolor="yellow", linewidth=1.8
    )
    ax_map.add_patch(outline)

    ax_truth = fig.add_subplot(gs_bottom[0, 1])
    ax_truth.set_title(f"{polygon} — ground truth", fontsize=10)
    ax_truth.set_axis_off()
    ax_truth.imshow(truth_rgb)

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

        for b, band in enumerate(cfg.bands):
            panels[band].set_data(render_band(inputs[b], valid, band))
        panels["pred"].set_data(masked_label_rgb(pred_ids, valid))
        map_artist.set_data(class_overlay(backdrop, acc.class_map()))
        outline.set_xy(tile_outline_px(r, acc))
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
        outline.set_visible(False)
        fig.suptitle(fig._suptitle.get_text() + "   (done — close window to exit)")
        plt.ioff()
        plt.show()


if __name__ == "__main__":
    main()
