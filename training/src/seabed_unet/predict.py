"""Prediction CLI: classified seabed map for one polygon.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_unet.predict \
      --config training/config/experiment_3band.yaml --polygon polygon4

Runs the model over the polygon's base (_rot) tiles, blends the 50%-overlapping
softmax fields (mask-weighted average, then argmax), and writes a North-up
classified GeoTIFF (uint8 class ids, 0 = no prediction) plus a colorized JPEG
with .jgw/.prj sidecars to <run_dir>/maps/.
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

from seabed_tiler.viz import label_to_rgb, write_prj, write_worldfile

from .config import load_config
from .data import load_run_records
from .inference import load_checkpoint, predict_probs
from .logging_utils import add_file_handler, setup_logging
from .normalize import apply_stats, compute_band_stats, load_stats
from .train import resolve_device

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def _records_extent(records) -> tuple[float, float, float, float, float]:
    """Axis-aligned envelope (xmin, ymin, xmax, ymax, res) of all (rotated) tiles."""
    xmin = ymin = math.inf
    xmax = ymax = -math.inf
    res = None
    for r in records:
        h, w = r.label.shape
        corners = [r.transform * c for c in [(0, 0), (w, 0), (0, h), (w, h)]]
        xs, ys = zip(*corners)
        xmin, xmax = min(xmin, *xs), max(xmax, *xs)
        ymin, ymax = min(ymin, *ys), max(ymax, *ys)
        if res is None:
            res = math.hypot(r.transform.a, r.transform.d)
    return xmin, ymin, xmax, ymax, res


def build_class_map(
    records, model, stats, bands, class_ids, nodata, band_modes, device
) -> tuple[np.ndarray, object, object]:
    """(uint8 class-id map, North-up transform, crs) from mask-weighted blended probs."""
    xmin, ymin, xmax, ymax, res = _records_extent(records)
    n_cols = int(round((xmax - xmin) / res))
    n_rows = int(round((ymax - ymin) / res))
    dst_transform = from_origin(xmin, ymax, res, res)
    crs = records[0].crs

    num_classes = len(class_ids)
    prob_sum = np.zeros((num_classes, n_rows, n_cols), dtype=np.float64)
    weight = np.zeros((n_rows, n_cols), dtype=np.float64)

    for r in records:
        inputs = apply_stats(r.features, r.polygon, bands, stats, nodata, band_modes)
        probs = predict_probs(model, inputs, device)
        # Mask-weighted contribution: pixels with invalid features carry no vote,
        # and the mask also kills reproject edge fill (outside-tile -> 0 weight).
        valid = (
            np.all(r.features != nodata, axis=0) & ~np.any(np.isnan(r.features), axis=0)
        ).astype(np.float32)
        src_stack = np.concatenate([probs * valid, valid[np.newaxis]], axis=0)
        dst_stack = np.zeros((num_classes + 1, n_rows, n_cols), dtype=np.float32)
        for b in range(num_classes + 1):
            reproject(
                source=src_stack[b],
                destination=dst_stack[b],
                src_transform=r.transform,
                src_crs=r.crs,
                dst_transform=dst_transform,
                dst_crs=crs,
                resampling=Resampling.bilinear,
            )
        prob_sum += dst_stack[:num_classes]
        weight += dst_stack[num_classes]

    covered = weight > 1e-3
    class_map = np.zeros((n_rows, n_cols), dtype=np.uint8)
    pred_channel = (prob_sum / np.maximum(weight, 1e-9)).argmax(axis=0)
    id_lookup = np.array(class_ids, dtype=np.uint8)
    class_map[covered] = id_lookup[pred_channel[covered]]
    return class_map, dst_transform, crs


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict a classified seabed map.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--checkpoint", default=None, help="Default: <run_dir>/best.pt")
    parser.add_argument(
        "--polygon", default=None,
        help="Polygon to map (default: first test polygon). May be a polygon "
             "unseen at training time — it self-normalizes in per_polygon mode.",
    )
    args = parser.parse_args(argv)

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)
    setup_logging()
    add_file_handler(cfg.run_dir / "predict.log")
    device = resolve_device(cfg.train.device)
    # Default target: first test polygon (polygon mode) or first split polygon
    # (spatial_blocks mode, where every polygon has a test region).
    fallback = cfg.split.test[0] if cfg.split.test else cfg.split.polygons[0]
    polygon = args.polygon or fallback

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
    logger.info(f"[+] {cfg.name}: mapping {polygon} from {len(records)} base tiles")

    stats = load_stats(cfg.run_dir / "normalization_stats.json")
    band_modes = cfg.normalization.modes_for(cfg.bands)
    per_poly_bands = [(i, b) for i, b in enumerate(cfg.bands) if band_modes[b] == "per_polygon"]
    if per_poly_bands and polygon not in stats:
        # Unseen survey: self-normalize its per-polygon bands from its own
        # features (no labels involved). Global bands keep the saved
        # train-fitted range — out-of-range values clip.
        arrays = [r.features for r in records]
        stats = dict(stats)
        stats[polygon] = {
            band: compute_band_stats(
                arrays, i, cfg.feature_nodata, tuple(cfg.normalization.clip_percentiles)
            )
            for i, band in per_poly_bands
        }
        logger.info(f"    {polygon} not in training stats — self-normalized "
              f"{[b for _, b in per_poly_bands]}")

    class_map, transform, crs = build_class_map(
        records, model, stats, cfg.bands, cfg.class_ids, cfg.feature_nodata,
        band_modes, device,
    )

    out_dir = cfg.run_dir / "maps"
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_path = out_dir / f"{polygon}_pred.tif"
    profile = {
        "driver": "GTiff", "height": class_map.shape[0], "width": class_map.shape[1],
        "count": 1, "dtype": "uint8", "crs": crs, "transform": transform,
        "nodata": 0, "compress": "deflate",
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(class_map, 1)

    from PIL import Image

    jpg_path = out_dir / f"{polygon}_pred.jpg"
    Image.fromarray(label_to_rgb(class_map)).save(jpg_path, quality=90)
    write_worldfile(jpg_path, transform)
    write_prj(jpg_path, crs)

    covered_px = int((class_map > 0).sum())
    logger.info(f"    {class_map.shape[0]}x{class_map.shape[1]} px, {covered_px:,} classified")
    logger.info(f"[+] map -> {tif_path} (+ colorized JPEG)")


if __name__ == "__main__":
    main()
