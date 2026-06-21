"""Offline "watch" recorder: replay every model classifying a polygon tile by tile,
but write compact static assets + JSON instead of drawing a matplotlib window.

The website (webapp/) plays these back so the "live" classification animation needs
NO inference at request time — torch/sklearn/rasterio run here, once per polygon.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_unet.export \
      --polygon polygon4 --polygon polygon3 --polygon polygon5

Per polygon it writes webapp/data/sessions/<polygon>/: a grayscale survey backdrop, a
ground-truth overlay, and per tile step the current tile's input bands + each model's
cumulative class map + tile classification, all driven by ONE shared tile sequence so
the models advance in lockstep. It then (re)writes webapp/data/catalog.json from every
manifest on disk. Reuses the exact helpers behind seabed_unet.watch / seabed_forest.watch,
so a recorded frame matches the desktop viewer.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from seabed_tiler.viz import LABEL_COLORS

from .config import load_config
from .data import IGNORE_INDEX, encode_target, load_run_records
from .inference import load_checkpoint, predict_probs
from .logging_utils import add_file_handler, remove_handler, setup_logging
from .metrics import confusion_matrix, dice_per_class
from .normalize import apply_stats, load_stats
from .predict import MapAccumulator, feature_valid_mask, resolve_polygon_stats
from .train import resolve_device
from .watch import (
    build_backdrop,
    build_truth_map,
    masked_label_rgb,
    render_band,
    tile_macro_dice,
    tile_outline_px,
)

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)

# Web-facing lane ids + labels. Tree models emit a raw and a guided-spatial lane.
LANE_LABELS = {
    "unet_3band": "U-Net (3-band)",
    "unet_2band": "U-Net (2-band)",
    "rf_raw": "Random Forest",
    "rf_spatial": "Random Forest (guided)",
    "hgb_raw": "HistGradBoost",
    "hgb_spatial": "HistGradBoost (guided)",
}
TREE_WEB_ID = {"random_forest": "rf", "hist_gradient_boosting": "hgb"}
OVERLAY_ALPHA = 0.8  # matches seabed_unet.watch.OVERLAY_ALPHA / class_overlay default


@dataclass
class ModelSource:
    """One model feeding the lockstep loop. ``tile_fields(record) -> (probs, guide)``
    normalizes internally with this model's own stats. A tree source with
    ``emit_spatial`` also produces a guided-regularized lane from the same accumulator."""

    web_id: str                 # "unet_3band", "rf", "hgb", ...
    label: str
    kind: str                   # "unet" | "tree"
    records: list               # tile records this source reads (its own band set)
    acc: MapAccumulator         # or PosteriorMapAccumulator for trees
    tile_fields: Callable
    emit_spatial: bool = False
    spatial_params: tuple[int, float] | None = None  # (radius, eps)
    dices: list[float] = field(default_factory=list)


def _overlay_rgba(class_map: np.ndarray) -> np.ndarray:
    """(H, W, 4) uint8: class colors (LABEL_COLORS) opaque, transparent where unclassified."""
    h, w = class_map.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    for cid, (r, g, b) in LABEL_COLORS.items():
        if cid == 0:
            continue
        m = class_map == cid
        rgba[m, 0], rgba[m, 1], rgba[m, 2], rgba[m, 3] = r, g, b, 255
    return rgba


def _save_overlay(class_map: np.ndarray, path: Path, out_size: tuple[int, int]) -> None:
    """Write a class-id map as an RGBA overlay PNG, NEAREST-resized to out_size (w, h)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(_overlay_rgba(class_map), "RGBA")
    if img.size != out_size:
        img = img.resize(out_size, Image.NEAREST)
    img.save(path)


def _save_rgb(rgb: np.ndarray, path: Path, fmt: str = "JPEG") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(rgb)
    img.save(path, quality=88) if fmt == "JPEG" else img.save(path)


def _map_dice(
    pred_class_map: np.ndarray, truth_map: np.ndarray, class_ids: list[int],
    id_to_name: dict[int, str],
) -> tuple[float | None, dict[str, float | None]]:
    """Macro + per-class Dice of a cumulative class map vs the truth map, over pixels that
    are both labeled and classified. Reuses the project confusion-matrix / Dice in channel
    space (so raw vs guided-spatial maps get genuinely different scores)."""
    C = len(class_ids)
    lut = np.full(int(max(class_ids)) + 1, IGNORE_INDEX, dtype=np.int64)
    for ch, cid in enumerate(class_ids):
        lut[cid] = ch
    t = lut[truth_map]
    p = lut[pred_class_map]
    t = np.where((t >= 0) & (p >= 0), t, IGNORE_INDEX)  # confusion_matrix drops target==IGNORE
    cm = confusion_matrix(p, t, C)
    if cm.sum() == 0:
        return None, {id_to_name[cid]: None for cid in class_ids}
    dice = dice_per_class(cm)
    per_class = {
        id_to_name[cid]: (None if np.isnan(dice[ch]) else float(dice[ch]))
        for ch, cid in enumerate(class_ids)
    }
    return float(np.nanmean(dice)), per_class


def _build_unet_source(
    config_path: Path, web_id: str, polygon: str, base: Path, device,
    records: list | None = None,
) -> ModelSource | None:
    """U-Net lane: own checkpoint + own normalization. ``records`` may be reused when the
    band set matches the primary (3-band) records; otherwise loaded for this band set."""
    cfg = load_config(config_path, base_dir=base)
    ckpt_path = cfg.run_dir / "best.pt"
    if not ckpt_path.exists():
        logger.warning(f"    skip {web_id}: no checkpoint at {ckpt_path}")
        return None
    model, ckpt = load_checkpoint(ckpt_path, device)
    if ckpt["config"]["bands"] != cfg.bands:
        raise SystemExit(f"{web_id}: checkpoint bands {ckpt['config']['bands']} != config {cfg.bands}")
    if records is None:
        records = load_run_records(cfg.rot_dir(polygon), polygon, cfg.bands, cfg.base_dir, augmented=False)
    band_modes = cfg.normalization.modes_for(cfg.bands)
    stats = resolve_polygon_stats(
        cfg, polygon, records, load_stats(cfg.run_dir / "normalization_stats.json"), band_modes
    )

    def tile_fields(r):
        inputs = apply_stats(r.features, r.polygon, cfg.bands, stats, cfg.feature_nodata, band_modes)
        return predict_probs(model, inputs, device), None

    return ModelSource(
        web_id=web_id, label=LANE_LABELS[web_id], kind="unet", records=records,
        acc=MapAccumulator(records, cfg.class_ids, cfg.feature_nodata), tile_fields=tile_fields,
    )


def _build_tree_sources(cfg, forest, polygon: str, records: list, stats: dict) -> list[ModelSource]:
    """One source per tree kind, normalizing with the forest stats; guided-spatial lane
    enabled when forest.spatial.enabled. Mirrors seabed_forest.watch.build_tree_families."""
    from seabed_forest.model import load_model
    from seabed_forest.predict import PosteriorMapAccumulator, _proba_field

    band_modes = cfg.normalization.modes_for(cfg.bands)
    spatial = forest.spatial.enabled
    guide_idx = cfg.bands.index(forest.spatial.guide_band) if spatial else None
    sources: list[ModelSource] = []
    for kind in forest.models:
        path = cfg.run_dir / f"model_{kind}.joblib"
        if not path.exists():
            logger.warning(f"    skip {kind}: no model at {path}")
            continue
        est = load_model(path)

        def tile_fields(r, est=est, gi=guide_idx):
            inputs = apply_stats(r.features, r.polygon, cfg.bands, stats, cfg.feature_nodata, band_modes)
            probs = _proba_field(est, inputs, cfg.num_classes)
            return probs, (inputs[gi] if gi is not None else None)

        sources.append(ModelSource(
            web_id=TREE_WEB_ID.get(kind, kind),
            label=LANE_LABELS.get(f"{TREE_WEB_ID.get(kind, kind)}_raw", kind),
            kind="tree", records=records,
            acc=PosteriorMapAccumulator(records, cfg.class_ids, cfg.feature_nodata),
            tile_fields=tile_fields, emit_spatial=spatial,
            spatial_params=(forest.spatial.radius, forest.spatial.eps) if spatial else None,
        ))
    return sources


def record_polygon(
    base: Path, polygon: str, *, forest_config: Path, unet_configs: dict[str, Path],
    data_dir: Path, max_long_side: int = 1400, device=None, max_tiles: int | None = None,
    models: list[str] | None = None,
) -> Path:
    """Record every available model classifying ``polygon`` tile by tile. Returns the
    written manifest path. The forest config drives the shared grid / backdrop / truth /
    band-display normalization (3 bands); U-Net lanes carry their own configs."""
    from seabed_forest.config import load_forest_config
    from seabed_forest.predict import PosteriorMapAccumulator, class_map_from_prob
    from seabed_forest.spatial import regularize_posterior

    cfg, forest = load_forest_config(forest_config, base_dir=base)
    class_ids, id_to_name = cfg.class_ids, cfg.id_to_name
    nodata, ignore_label, n_classes = cfg.feature_nodata, cfg.ignore_label, cfg.num_classes
    id_lookup = np.array(class_ids, dtype=np.uint8)

    main_records = load_run_records(cfg.rot_dir(polygon), polygon, cfg.bands, cfg.base_dir, augmented=False)
    if not main_records:
        raise SystemExit(f"no tiles found for {polygon} under {cfg.rot_dir(polygon)}")
    if max_tiles is not None:
        main_records = main_records[:max_tiles]
    main_band_modes = cfg.normalization.modes_for(cfg.bands)
    main_stats = resolve_polygon_stats(
        cfg, polygon, main_records, load_stats(cfg.run_dir / "normalization_stats.json"), main_band_modes
    )
    grid_acc = MapAccumulator(main_records, class_ids, nodata)
    n_tiles = len(main_records)
    if device is None:
        device = resolve_device("auto")

    logger.info(f"[+] recording {polygon}: {n_tiles} tiles, grid {grid_acc.n_rows}x{grid_acc.n_cols}")

    # --- build model sources (skip any whose artifacts are missing) ---
    sources: list[ModelSource] = []
    sources += _build_tree_sources(cfg, forest, polygon, main_records, main_stats)
    u3 = _build_unet_source(unet_configs["unet_3band"], "unet_3band", polygon, base, device,
                            records=main_records)  # same 3 bands -> reuse records
    if u3:
        sources.append(u3)
    if "unet_2band" in unet_configs:
        u2 = _build_unet_source(unet_configs["unet_2band"], "unet_2band", polygon, base, device)
        if u2 and len(u2.records) == n_tiles:
            sources.append(u2)
        elif u2:
            logger.warning(f"    skip unet_2band: {len(u2.records)} tiles != {n_tiles}")

    # lane ids each source emits (raw + optional spatial), narrowed by the --models filter
    keep = set(models) if models is not None else None

    def lane_ids(src: ModelSource) -> list[str]:
        if src.kind == "tree":
            ids = [f"{src.web_id}_raw"]
            if src.emit_spatial:
                ids.append(f"{src.web_id}_spatial")
        else:
            ids = [src.web_id]
        return [lane for lane in ids if keep is None or lane in keep]

    sources = [s for s in sources if lane_ids(s)]
    if not sources:
        raise SystemExit(f"{polygon}: no model sources available to record")
    for s in sources:
        for acc in (s.acc,):
            if (acc.n_rows, acc.n_cols) != (grid_acc.n_rows, grid_acc.n_cols):
                raise SystemExit(f"{s.web_id}: grid {acc.n_rows}x{acc.n_cols} != {grid_acc.n_rows}x{grid_acc.n_cols}")

    # --- output geometry (downscale the map grid to a sane web size) ---
    scale = min(1.0, max_long_side / max(grid_acc.n_rows, grid_acc.n_cols))
    out_w = max(1, round(grid_acc.n_cols * scale))
    out_h = max(1, round(grid_acc.n_rows * scale))
    sx, sy = out_w / grid_acc.n_cols, out_h / grid_acc.n_rows
    out_size = (out_w, out_h)

    session = data_dir / "sessions" / polygon
    steps_dir = session / "steps"
    (steps_dir).mkdir(parents=True, exist_ok=True)

    logger.info(f"    building backdrop + truth ({n_tiles} tiles) ...")
    backdrop = build_backdrop(main_records, grid_acc, cfg.bands, main_stats, main_band_modes, nodata)
    Image.fromarray(backdrop).resize(out_size, Image.BILINEAR).save(session / "backdrop.jpg", quality=88)
    truth_ids = build_truth_map(main_records, grid_acc)
    _save_overlay(truth_ids, session / "truth.png", out_size)

    pad = max(4, len(str(n_tiles)))
    final_maps: dict[str, np.ndarray] = {}  # lane_id -> final class map (for summary)
    steps: list[dict] = []

    for i in range(n_tiles):
        idx = f"{i + 1:0{pad}d}"
        main_r = main_records[i]
        valid_main = feature_valid_mask(main_r.features, nodata)
        display = apply_stats(main_r.features, main_r.polygon, cfg.bands, main_stats, nodata, main_band_modes)
        band_files: dict[str, str] = {}
        for b, band in enumerate(cfg.bands):
            rel = f"steps/{idx}_{band}.jpg"
            _save_rgb(render_band(display[b], valid_main, band), session / rel, "JPEG")
            band_files[band] = rel
        outline = [[round(x * sx, 2), round(y * sy, 2)] for x, y in tile_outline_px(main_r, grid_acc)]

        per_model: dict[str, dict] = {}
        for src in sources:
            r = src.records[i]
            probs, guide = src.tile_fields(r)
            if isinstance(src.acc, PosteriorMapAccumulator):
                src.acc.add(r, probs, guide)
            else:
                src.acc.add(r, probs)
            valid = feature_valid_mask(r.features, nodata)
            pred_channels = probs.argmax(axis=0)
            target = encode_target(r.label, r.features, class_ids, nodata, ignore_label)
            dice = tile_macro_dice(pred_channels, target, n_classes)
            if not np.isnan(dice):
                src.dices.append(dice)
            running = float(np.mean(src.dices)) if src.dices else None
            pred_rgb = masked_label_rgb(id_lookup[pred_channels], valid)

            for lane in lane_ids(src):
                if lane.endswith("_spatial"):
                    radius, eps = src.spatial_params
                    prob, gmap, covered = src.acc.posterior()
                    cmap = class_map_from_prob(
                        regularize_posterior(prob, gmap, covered, radius, eps), covered, class_ids
                    )
                else:
                    cmap = src.acc.class_map()
                _save_overlay(cmap, session / f"steps/{lane}/{idx}_map.png", out_size)
                _save_rgb(pred_rgb, session / f"steps/{lane}/{idx}_pred.png", "PNG")
                per_model[lane] = {
                    "class_map": f"steps/{lane}/{idx}_map.png",
                    "pred": f"steps/{lane}/{idx}_pred.png",
                    "tile_dice": None if np.isnan(dice) else round(float(dice), 4),
                    "running_mean_dice": None if running is None else round(running, 4),
                }
                if i == n_tiles - 1:
                    final_maps[lane] = cmap

        steps.append({
            "i": i + 1, "tile_id": main_r.tile_id, "outline_px": outline,
            "bands": band_files, "per_model": per_model,
        })
        logger.info(f"    tile {i + 1}/{n_tiles} {main_r.tile_id}")

    # --- per-lane summary (map-level macro Dice vs ground truth) ---
    model_entries: list[dict] = []
    for src in sources:
        for lane in lane_ids(src):
            macro, per_class = _map_dice(final_maps[lane], truth_ids, class_ids, id_to_name)
            model_entries.append({
                "id": lane, "label": LANE_LABELS.get(lane, lane), "kind": src.kind,
                "spatial": lane.endswith("_spatial"),
                "summary": {
                    "mean_dice": None if macro is None else round(macro, 4),
                    "per_class_dice": {k: (None if v is None else round(v, 4)) for k, v in per_class.items()},
                },
            })

    manifest = {
        "polygon": polygon,
        "bands": list(cfg.bands),
        "class_names": {str(cid): id_to_name[cid] for cid in class_ids},
        "class_palette": {str(cid): list(rgb) for cid, rgb in LABEL_COLORS.items()},
        "overlay_alpha": OVERLAY_ALPHA,
        "map_size": {"width": out_w, "height": out_h},
        "backdrop": "backdrop.jpg",
        "truth_map": "truth.png",
        "n_tiles": n_tiles,
        "models": model_entries,
        "steps": steps,
    }
    manifest_path = session / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info(f"[+] {polygon}: wrote {manifest_path} ({len(model_entries)} lanes, {n_tiles} steps)")
    return manifest_path


def rebuild_catalog(data_dir: Path) -> Path:
    """(Re)write catalog.json from every manifest under data_dir/sessions/."""
    sessions = sorted((data_dir / "sessions").glob("*/manifest.json"))
    polygons = []
    class_names: dict = {}
    class_palette: dict = {}
    for mpath in sessions:
        m = json.loads(mpath.read_text())
        class_names = m.get("class_names", class_names)
        class_palette = m.get("class_palette", class_palette)
        polygons.append({
            "polygon": m["polygon"],
            "n_tiles": m["n_tiles"],
            "manifest": f"sessions/{m['polygon']}/manifest.json",
            "models": [{"id": e["id"], "mean_dice": e["summary"]["mean_dice"]} for e in m["models"]],
        })
    catalog = {
        "class_names": class_names,
        "class_palette": class_palette,
        "model_labels": LANE_LABELS,
        "polygons": polygons,
    }
    path = data_dir / "catalog.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2))
    logger.info(f"[+] catalog -> {path} ({len(polygons)} polygon(s))")
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Record models classifying polygon(s) tile by tile for the web demo.")
    parser.add_argument("--polygon", action="append", required=True, help="Polygon to record (repeatable).")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--forest-config", default="training/config/forest_3band.yaml",
                        help="Forest config: drives grid/backdrop/truth + the tree lanes.")
    parser.add_argument("--unet-3band-config", default="training/config/experiment_3band.yaml")
    parser.add_argument("--unet-2band-config", default="training/config/experiment_2band.yaml")
    parser.add_argument("--data-dir", default="webapp/data", help="Output root (default: webapp/data).")
    parser.add_argument("--max-long-side", type=int, default=1400, help="Cap the map grid's long side (px).")
    parser.add_argument("--device", default=None, help="Override U-Net device (cpu/mps/cuda).")
    parser.add_argument("--max-tiles", type=int, default=None, help="Record only the first N tiles (smoke).")
    parser.add_argument("--models", default=None, help="Comma subset of lane ids to record (default: all available).")
    args = parser.parse_args(argv)

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    setup_logging()
    data_dir = (base / args.data_dir).resolve()
    handler = add_file_handler(data_dir / "export.log")
    try:
        device = resolve_device(args.device) if args.device else resolve_device("auto")
        unet_configs = {
            "unet_3band": Path(args.unet_3band_config),
            "unet_2band": Path(args.unet_2band_config),
        }
        models = [m.strip() for m in args.models.split(",")] if args.models else None
        for polygon in args.polygon:
            record_polygon(
                base, polygon, forest_config=Path(args.forest_config), unet_configs=unet_configs,
                data_dir=data_dir, max_long_side=args.max_long_side, device=device,
                max_tiles=args.max_tiles, models=models,
            )
        rebuild_catalog(data_dir)
    finally:
        remove_handler(handler)


if __name__ == "__main__":
    main()
