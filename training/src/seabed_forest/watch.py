"""Live inference viewer for the tree baseline: watch RF + HistGradientBoosting
classify a polygon tile by tile, raw vs guided-spatial side by side.

Run from repo root (needs a GUI backend — a normal desktop terminal, NOT Agg):
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_forest.watch \
      --config training/config/forest_3band.yaml --polygon polygon4

Top row: the current tile's input bands + each model's classification of that tile.
Bottom row: one full-polygon map per (model x {raw, spatial}) plus ground truth, all
filling in as tiles are classified (same mask-weighted blending as seabed_forest.predict).
The spatial map is re-regularized every tile (guided filter). --save writes a half-res GIF.
Mirrors seabed_unet.watch and reuses its helpers; only the prediction is tree-based.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

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


def _resolve_kinds(arg_models: str | None, forest) -> list[str]:
    if not arg_models:
        return list(forest.models)
    kinds = [m.strip() for m in arg_models.split(",") if m.strip()]
    bad = [m for m in kinds if m not in SUPPORTED_MODELS]
    if bad:
        raise SystemExit(f"--models: unknown kind(s) {bad}; valid: {list(SUPPORTED_MODELS)}")
    return kinds


def run_watch(cfg, forest, polygon, kinds, spatial=True, delay=0.4, save=False,
              max_tiles=None, block=True) -> list:
    """Drive the live viewer. Returns captured frames (empty unless save=True)."""
    band_modes = cfg.normalization.modes_for(cfg.bands)
    if spatial and forest.spatial.guide_band not in cfg.bands:
        raise SystemExit(
            f"spatial.guide_band {forest.spatial.guide_band!r} not in bands {cfg.bands}"
        )
    guide_idx = cfg.bands.index(forest.spatial.guide_band) if spatial else None

    records = load_run_records(
        cfg.rot_dir(polygon), polygon, cfg.bands, cfg.base_dir, augmented=False
    )
    if not records:
        raise SystemExit(f"no tiles found for {polygon}")
    if max_tiles is not None:
        records = records[:max_tiles]

    stats = resolve_polygon_stats(
        cfg, polygon, records, load_stats(cfg.run_dir / "normalization_stats.json"), band_modes
    )
    logger.info(f"[+] {cfg.name}: watching {polygon} — {len(records)} tiles, "
                f"models {kinds} (RandomForest can be ~GB to load) ...")
    estimators = {k: load_model(cfg.run_dir / f"model_{k}.joblib") for k in kinds}
    accs = {k: PosteriorMapAccumulator(records, cfg.class_ids, cfg.feature_nodata) for k in kinds}
    id_lookup = np.array(cfg.class_ids, dtype=np.uint8)
    grid_acc = accs[kinds[0]]

    logger.info(f"    building survey backdrop + truth map ({len(records)} tiles) ...")
    backdrop = build_backdrop(records, grid_acc, cfg.bands, stats, band_modes, cfg.feature_nodata)
    truth_rgb = class_overlay(backdrop, build_truth_map(records, grid_acc))

    variants: list[tuple[str, str]] = []
    for k in kinds:
        variants.append((k, "raw"))
        if spatial:
            variants.append((k, "spatial"))

    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MplPolygon

    plt.ion()
    fig = plt.figure(figsize=(4 + 3 * len(variants), 9))
    if fig.canvas.manager is not None:
        fig.canvas.manager.set_window_title(f"seabed_forest — {cfg.name} / {polygon}")
    n_top = len(cfg.bands) + len(kinds)
    n_bottom = len(variants) + 1
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 2.2])
    gs_top = outer[0].subgridspec(1, n_top)
    gs_bottom = outer[1].subgridspec(1, n_bottom)

    band_panels = {}
    for i, band in enumerate(cfg.bands):
        ax = fig.add_subplot(gs_top[0, i])
        ax.set_title(f"tile · {band}", fontsize=9)
        ax.set_axis_off()
        band_panels[band] = ax.imshow(np.zeros((2, 2, 3), np.uint8))
    pred_panels = {}
    for j, k in enumerate(kinds):
        ax = fig.add_subplot(gs_top[0, len(cfg.bands) + j])
        ax.set_title(f"tile · {MODEL_LABELS.get(k, k)}", fontsize=9)
        ax.set_axis_off()
        pred_panels[k] = ax.imshow(np.zeros((2, 2, 3), np.uint8))

    map_artists = {}
    outlines = []
    for c, (k, variant) in enumerate(variants):
        ax = fig.add_subplot(gs_bottom[0, c])
        ax.set_title(f"{MODEL_LABELS.get(k, k)} · {variant}", fontsize=9)
        ax.set_axis_off()
        map_artists[(k, variant)] = ax.imshow(backdrop)
        ol = MplPolygon(np.zeros((4, 2)), closed=True, fill=False,
                        edgecolor="yellow", linewidth=1.6)
        ax.add_patch(ol)
        outlines.append(ol)
    ax_truth = fig.add_subplot(gs_bottom[0, n_bottom - 1])
    ax_truth.set_title(f"{polygon} · ground truth", fontsize=9)
    ax_truth.set_axis_off()
    ax_truth.imshow(truth_rgb)

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.show(block=False)

    frames: list = []
    dices = {k: [] for k in kinds}
    radius, eps = forest.spatial.radius, forest.spatial.eps
    t0 = time.monotonic()
    for i, r in enumerate(records, start=1):
        inputs = apply_stats(r.features, r.polygon, cfg.bands, stats, cfg.feature_nodata, band_modes)
        valid = feature_valid_mask(r.features, cfg.feature_nodata)
        target = encode_target(r.label, r.features, cfg.class_ids, cfg.feature_nodata, cfg.ignore_label)
        for b, band in enumerate(cfg.bands):
            band_panels[band].set_data(render_band(inputs[b], valid, band))

        tile_titles = []
        for k in kinds:
            probs = _proba_field(estimators[k], inputs, cfg.num_classes)
            accs[k].add(r, probs, inputs[guide_idx] if spatial else None)
            pred_channels = probs.argmax(axis=0)
            pred_panels[k].set_data(masked_label_rgb(id_lookup[pred_channels], valid))
            dice = tile_macro_dice(pred_channels, target, cfg.num_classes)
            if not np.isnan(dice):
                dices[k].append(dice)
            label = MODEL_LABELS.get(k, k)
            tile_titles.append(f"{label} {'n/a' if np.isnan(dice) else f'{dice:.2f}'}")
            map_artists[(k, "raw")].set_data(class_overlay(backdrop, accs[k].class_map()))
            if spatial:
                prob, gmap, covered = accs[k].posterior()
                reg = regularize_posterior(prob, gmap, covered, radius, eps)
                smap = class_map_from_prob(reg, covered, cfg.class_ids)
                map_artists[(k, "spatial")].set_data(class_overlay(backdrop, smap))

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
    means = {k: (float(np.mean(v)) if v else float("nan")) for k, v in dices.items()}
    logger.info(f"[+] watched {len(records)} tiles in {elapsed:.1f}s — mean tile Dice " +
                ", ".join(f"{MODEL_LABELS.get(k, k)} {means[k]:.3f}" for k in kinds))

    if save and frames:
        from PIL import Image

        gif_path = cfg.run_dir / "maps" / f"{polygon}_watch_forest.gif"
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
    parser = argparse.ArgumentParser(description="Watch the seabed tree baseline classify tiles live.")
    parser.add_argument("--config", required=True, help="Forest experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--polygon", default=None, help="Default: first test/split polygon.")
    parser.add_argument("--delay", type=float, default=0.4, help="Seconds per tile.")
    parser.add_argument("--models", default=None,
                        help="Comma subset of model kinds to show (default: config forest.models).")
    parser.add_argument("--no-spatial", action="store_true", help="Drop the spatial columns.")
    parser.add_argument("--save", action="store_true",
                        help="Also write <run_dir>/maps/<polygon>_watch_forest.gif")
    args = parser.parse_args(argv)

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg, forest = load_forest_config(args.config, base_dir=base)
    setup_logging()
    add_file_handler(cfg.run_dir / "watch_forest.log")
    polygon = args.polygon or (cfg.split.test[0] if cfg.split.test else cfg.split.polygons[0])
    kinds = _resolve_kinds(args.models, forest)
    run_watch(cfg, forest, polygon, kinds, spatial=not args.no_spatial,
              delay=args.delay, save=args.save)


if __name__ == "__main__":
    main()
