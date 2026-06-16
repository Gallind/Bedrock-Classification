"""Live multi-model viewer: watch RF + HistGradientBoosting (raw vs guided-spatial)
AND the U-Net classify a polygon tile by tile, all side by side.

Run from repo root (needs a GUI backend — a normal desktop terminal, NOT Agg):
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_forest.watch \
      --config training/config/forest_3band.yaml --polygon polygon4

A "model family" (RF / HGB / U-Net) carries its OWN normalization and predict step, so
heterogeneous models — sklearn trees and the torch U-Net — share one live view, each
seeing inputs exactly as it was trained. Top row: the current tile's input bands + each
family's classification of that tile. Bottom grid (wrapped to <=3 columns): one full-polygon
map per (family x {raw, spatial}) plus ground truth, filling in live; tree spatial maps are
re-regularized every tile. --save writes a half-res GIF. Reuses seabed_unet.watch helpers.
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from seabed_unet.data import encode_target, load_run_records
from seabed_unet.logging_utils import add_file_handler, setup_logging
from seabed_unet.normalize import apply_stats, load_stats
from seabed_unet.predict import feature_valid_mask, resolve_polygon_stats
from seabed_unet.watch import (
    build_backdrop,
    build_truth_map,
    class_overlay,
    masked_label_rgb,
    render_band,
    tile_macro_dice,
    tile_outline_px,
)

from .config import SUPPORTED_MODELS, load_forest_config
from .model import load_model
from .predict import PosteriorMapAccumulator, _proba_field, class_map_from_prob
from .spatial import regularize_posterior

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)

MODEL_LABELS = {"random_forest": "RF", "hist_gradient_boosting": "HGB"}


@dataclass
class ModelFamily:
    """One model shown in the viewer. ``tile_fields`` maps a tile record to its
    (probs (C,H,W), guide (H,W) or None), normalizing internally with this model's own
    stats. ``supports_spatial`` adds a guided-regularized map column for this family."""

    label: str
    acc: PosteriorMapAccumulator
    tile_fields: Callable
    supports_spatial: bool


def _resolve_kinds(arg_models: str | None, forest) -> list[str]:
    if not arg_models:
        return list(forest.models)
    kinds = [m.strip() for m in arg_models.split(",") if m.strip()]
    bad = [m for m in kinds if m not in SUPPORTED_MODELS]
    if bad:
        raise SystemExit(f"--models: unknown kind(s) {bad}; valid: {list(SUPPORTED_MODELS)}")
    return kinds


def build_tree_families(cfg, forest, kinds, spatial, records, stats) -> list[ModelFamily]:
    """One family per tree kind, normalizing with the forest stats; spatial guide =
    forest-normalized guide band."""
    band_modes = cfg.normalization.modes_for(cfg.bands)
    guide_idx = cfg.bands.index(forest.spatial.guide_band) if spatial else None
    families = []
    for k in kinds:
        est = load_model(cfg.run_dir / f"model_{k}.joblib")

        def tile_fields(r, est=est, gi=guide_idx):
            inputs = apply_stats(r.features, r.polygon, cfg.bands, stats, cfg.feature_nodata, band_modes)
            probs = _proba_field(est, inputs, cfg.num_classes)
            guide = inputs[gi] if gi is not None else None
            return probs, guide

        families.append(ModelFamily(
            label=MODEL_LABELS.get(k, k),
            acc=PosteriorMapAccumulator(records, cfg.class_ids, cfg.feature_nodata),
            tile_fields=tile_fields,
            supports_spatial=spatial,
        ))
    return families


def build_unet_family(unet_cfg, records, polygon, device) -> ModelFamily:
    """U-Net family: own checkpoint + own normalization; raw only (no spatial column)."""
    from seabed_unet.inference import load_checkpoint, predict_probs

    ckpt_path = unet_cfg.run_dir / "best.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no U-Net checkpoint at {ckpt_path} — train it or pass --no-unet")
    model, ckpt = load_checkpoint(ckpt_path, device)
    if ckpt["config"]["bands"] != unet_cfg.bands:
        raise SystemExit(f"U-Net checkpoint bands {ckpt['config']['bands']} != config {unet_cfg.bands}")
    band_modes = unet_cfg.normalization.modes_for(unet_cfg.bands)
    stats = resolve_polygon_stats(
        unet_cfg, polygon, records,
        load_stats(unet_cfg.run_dir / "normalization_stats.json"), band_modes,
    )

    def tile_fields(r):
        inputs = apply_stats(r.features, r.polygon, unet_cfg.bands, stats, unet_cfg.feature_nodata, band_modes)
        return predict_probs(model, inputs, device), None

    return ModelFamily(
        label="U-Net",
        acc=PosteriorMapAccumulator(records, unet_cfg.class_ids, unet_cfg.feature_nodata),
        tile_fields=tile_fields,
        supports_spatial=False,
    )


def run_watch(cfg, forest, polygon, kinds, spatial=True, unet_cfg=None,
              delay=0.4, save=False, max_tiles=None, block=True, device=None) -> list:
    """Build the model families (trees + optional U-Net) and drive the viewer."""
    band_modes = cfg.normalization.modes_for(cfg.bands)
    if spatial and forest.spatial.guide_band not in cfg.bands:
        raise SystemExit(f"spatial.guide_band {forest.spatial.guide_band!r} not in bands {cfg.bands}")

    records = load_run_records(cfg.rot_dir(polygon), polygon, cfg.bands, cfg.base_dir, augmented=False)
    if not records:
        raise SystemExit(f"no tiles found for {polygon}")
    if max_tiles is not None:
        records = records[:max_tiles]

    stats = resolve_polygon_stats(
        cfg, polygon, records, load_stats(cfg.run_dir / "normalization_stats.json"), band_modes
    )
    logger.info(f"[+] {cfg.name}: watching {polygon} — {len(records)} tiles, "
                f"trees {kinds}{' + U-Net' if unet_cfg is not None else ''} "
                f"(RandomForest can be ~GB to load) ...")
    families = build_tree_families(cfg, forest, kinds, spatial, records, stats)
    if unet_cfg is not None:
        if list(unet_cfg.bands) != list(cfg.bands):
            raise SystemExit(f"U-Net bands {unet_cfg.bands} must match forest bands {cfg.bands}")
        if unet_cfg.class_ids != cfg.class_ids:
            raise SystemExit(f"U-Net class_ids {unet_cfg.class_ids} must match {cfg.class_ids}")
        if device is None:
            from seabed_unet.train import resolve_device
            device = resolve_device(unet_cfg.train.device)
        families.append(build_unet_family(unet_cfg, records, polygon, device))

    return _watch_families(families, records, cfg, forest, stats, band_modes,
                           polygon, delay, save, block)


def _watch_families(families, records, cfg, forest, stats, band_modes,
                    polygon, delay, save, block) -> list:
    """Generalized figure + per-tile loop over heterogeneous model families.

    Layout: top row = input bands + each family's tile classification. Bottom = one
    COLUMN per family (raw on top, guided-spatial below when supported); ground truth
    takes the last family's free bottom slot (e.g. under the U-Net) or its own column.
    """
    grid_acc = families[0].acc
    id_lookup = np.array(cfg.class_ids, dtype=np.uint8)
    radius, eps = forest.spatial.radius, forest.spatial.eps

    logger.info(f"    building survey backdrop + truth map ({len(records)} tiles) ...")
    backdrop = build_backdrop(records, grid_acc, cfg.bands, stats, band_modes, cfg.feature_nodata)
    truth_rgb = class_overlay(backdrop, build_truth_map(records, grid_acc))

    # column-major: column c == family c (raw on row 0, spatial on row 1)
    has_spatial = any(f.supports_spatial for f in families)
    n_rows = 2 if has_spatial else 1
    n_cols = len(families)
    placements: dict[tuple[int, int], tuple[str, str]] = {}
    for c, fam in enumerate(families):
        placements[(0, c)] = (fam.label, "raw")
        if fam.supports_spatial:
            placements[(1, c)] = (fam.label, "spatial")
    if n_rows == 2 and (1, n_cols - 1) not in placements:
        truth_rc = (1, n_cols - 1)            # under the U-Net (its column has no spatial row)
    else:
        truth_rc = (0, n_cols)                # no free slot -> ground truth gets its own column
        n_cols += 1

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    plt.ion()
    n_top = len(cfg.bands) + len(families)
    fig = plt.figure(figsize=(3.2 * max(n_top, n_cols), 3.4 * (1 + n_rows)),
                     constrained_layout=True)
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f"seabed multi-model — {cfg.name} / {polygon}")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15 * n_rows])
    gs_top = outer[0].subgridspec(1, n_top)
    gs_bottom = outer[1].subgridspec(n_rows, n_cols)

    band_panels = {}
    for i, band in enumerate(cfg.bands):
        ax = fig.add_subplot(gs_top[0, i])
        ax.set_title(f"tile · {band}", fontsize=9)
        ax.set_axis_off()
        band_panels[band] = ax.imshow(np.zeros((2, 2, 3), np.uint8))
    pred_panels = {}
    for j, fam in enumerate(families):
        ax = fig.add_subplot(gs_top[0, len(cfg.bands) + j])
        ax.set_title(f"tile · {fam.label}", fontsize=9)
        ax.set_axis_off()
        pred_panels[fam.label] = ax.imshow(np.zeros((2, 2, 3), np.uint8))

    map_artists = {}
    outlines = []
    for (rr, cc), key in placements.items():
        ax = fig.add_subplot(gs_bottom[rr, cc])
        ax.set_title(f"{key[0]} · {key[1]}", fontsize=9)
        ax.set_axis_off()
        map_artists[key] = ax.imshow(backdrop)
        ol = MplPolygon(np.zeros((4, 2)), closed=True, fill=False, edgecolor="yellow", linewidth=1.6)
        ax.add_patch(ol)
        outlines.append(ol)
    ax_truth = fig.add_subplot(gs_bottom[truth_rc[0], truth_rc[1]])
    ax_truth.set_title(f"{polygon} · ground truth", fontsize=9)
    ax_truth.set_axis_off()
    ax_truth.imshow(truth_rgb)

    plt.show(block=False)

    frames: list = []
    dices = {fam.label: [] for fam in families}
    t0 = time.monotonic()
    for i, r in enumerate(records, start=1):
        display = apply_stats(r.features, r.polygon, cfg.bands, stats, cfg.feature_nodata, band_modes)
        valid = feature_valid_mask(r.features, cfg.feature_nodata)
        target = encode_target(r.label, r.features, cfg.class_ids, cfg.feature_nodata, cfg.ignore_label)
        for b, band in enumerate(cfg.bands):
            band_panels[band].set_data(render_band(display[b], valid, band))

        tile_titles = []
        for fam in families:
            probs, guide = fam.tile_fields(r)
            fam.acc.add(r, probs, guide)
            pred_channels = probs.argmax(axis=0)
            pred_panels[fam.label].set_data(masked_label_rgb(id_lookup[pred_channels], valid))
            dice = tile_macro_dice(pred_channels, target, cfg.num_classes)
            if not np.isnan(dice):
                dices[fam.label].append(dice)
            tile_titles.append(f"{fam.label} {'n/a' if np.isnan(dice) else f'{dice:.2f}'}")
            map_artists[(fam.label, "raw")].set_data(class_overlay(backdrop, fam.acc.class_map()))
            if fam.supports_spatial:
                prob, gmap, covered = fam.acc.posterior()
                reg = regularize_posterior(prob, gmap, covered, radius, eps)
                smap = class_map_from_prob(reg, covered, cfg.class_ids)
                map_artists[(fam.label, "spatial")].set_data(class_overlay(backdrop, smap))

        xy = tile_outline_px(r, grid_acc)
        for ol in outlines:
            ol.set_xy(xy)
        fig.suptitle(f"tile {i}/{len(records)} · {r.tile_id} · " + "  ".join(tile_titles), fontsize=11)
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(max(delay, 0.001))

        if save:
            buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
            frames.append(buf[::2, ::2].copy())

        logger.info(f"    tile {i}/{len(records)} {r.tile_id}  " + " ".join(tile_titles))
        if not plt.fignum_exists(fig.number):
            logger.info("[+] window closed — stopping")
            break

    elapsed = time.monotonic() - t0
    means = {fam.label: (float(np.mean(dices[fam.label])) if dices[fam.label] else float("nan"))
             for fam in families}
    logger.info(f"[+] watched {len(records)} tiles in {elapsed:.1f}s — mean tile Dice " +
                ", ".join(f"{fam.label} {means[fam.label]:.3f}" for fam in families))

    if save and frames:
        from PIL import Image

        gif_path = cfg.run_dir / "maps" / f"{polygon}_watch_multi.gif"
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        images = [Image.fromarray(f) for f in frames]
        images[0].save(gif_path, save_all=True, append_images=images[1:],
                       duration=int(max(delay, 0.15) * 1000), loop=0)
        logger.info(f"[+] animation -> {gif_path}")

    if block and plt.fignum_exists(fig.number):
        for ol in outlines:
            ol.set_visible(False)
        fig.suptitle(fig._suptitle.get_text() + "   (done — close window to exit)")
        plt.ioff()
        plt.show()
    else:
        plt.close(fig)
    return frames


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Watch RF + HGB + U-Net classify tiles live.")
    parser.add_argument("--config", required=True, help="Forest experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--polygon", default=None, help="Default: first test/split polygon.")
    parser.add_argument("--delay", type=float, default=0.4, help="Seconds per tile.")
    parser.add_argument("--models", default=None,
                        help="Comma subset of TREE kinds to show (default: config forest.models).")
    parser.add_argument("--no-spatial", action="store_true", help="Drop the tree spatial columns.")
    parser.add_argument("--unet-config", default="training/config/experiment_3band.yaml",
                        help="U-Net experiment config (its run dir holds best.pt + stats).")
    parser.add_argument("--no-unet", action="store_true", help="Trees only; do not show the U-Net.")
    parser.add_argument("--save", action="store_true",
                        help="Also write <run_dir>/maps/<polygon>_watch_multi.gif")
    args = parser.parse_args(argv)

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg, forest = load_forest_config(args.config, base_dir=base)
    setup_logging()
    add_file_handler(cfg.run_dir / "watch_multi.log")
    polygon = args.polygon or (cfg.split.test[0] if cfg.split.test else cfg.split.polygons[0])
    kinds = _resolve_kinds(args.models, forest)

    unet_cfg = None
    if not args.no_unet:
        from seabed_unet.config import load_config
        unet_cfg = load_config(args.unet_config, base_dir=base)

    run_watch(cfg, forest, polygon, kinds, spatial=not args.no_spatial, unet_cfg=unet_cfg,
              delay=args.delay, save=args.save)


if __name__ == "__main__":
    main()
