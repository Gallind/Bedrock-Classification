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


def resolve_polygon_stats(cfg, polygon, records, stats, band_modes) -> dict:
    """Ensure normalization stats exist for ``polygon`` (shared with watch.py).

    Unseen survey: self-normalize its per-polygon bands from its own features
    (no labels involved). Global bands keep the saved train-fitted range —
    out-of-range values clip.
    """
    per_poly_bands = [(i, b) for i, b in enumerate(cfg.bands) if band_modes[b] == "per_polygon"]
    if not per_poly_bands or polygon in stats:
        return stats
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
    return stats


def feature_valid_mask(features: np.ndarray, nodata: float) -> np.ndarray:
    """(H, W) bool: True where every feature band has real data."""
    return np.all(features != nodata, axis=0) & ~np.any(np.isnan(features), axis=0)


class MapAccumulator:
    """Accumulates mask-weighted tile softmax fields into a North-up class map.

    Used in one shot by build_class_map() and incrementally by seabed_unet.watch
    (call class_map() after any number of add() calls to see the map so far).
    """

    def __init__(self, records, class_ids: list[int], nodata: float):
        if not records:
            raise ValueError("no tile records to accumulate")
        xmin, ymin, xmax, ymax, res = _records_extent(records)
        self.n_cols = int(round((xmax - xmin) / res))
        self.n_rows = int(round((ymax - ymin) / res))
        self.transform = from_origin(xmin, ymax, res, res)
        self.crs = records[0].crs
        self.class_ids = class_ids
        self.nodata = nodata
        self._prob_sum = np.zeros((len(class_ids), self.n_rows, self.n_cols), dtype=np.float64)
        self._weight = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)

    def add(self, record, probs: np.ndarray) -> None:
        """Blend one tile's (C, H, W) softmax field into the map.

        Mask-weighted contribution: pixels with invalid features carry no vote,
        and the mask also kills reproject edge fill (outside-tile -> 0 weight).
        """
        num_classes = len(self.class_ids)
        valid = feature_valid_mask(record.features, self.nodata).astype(np.float32)
        src_stack = np.concatenate([probs * valid, valid[np.newaxis]], axis=0)
        dst_stack = np.zeros((num_classes + 1, self.n_rows, self.n_cols), dtype=np.float32)
        for b in range(num_classes + 1):
            reproject(
                source=src_stack[b],
                destination=dst_stack[b],
                src_transform=record.transform,
                src_crs=record.crs,
                dst_transform=self.transform,
                dst_crs=self.crs,
                resampling=Resampling.bilinear,
            )
        self._prob_sum += dst_stack[:num_classes]
        self._weight += dst_stack[num_classes]

    def class_map(self) -> np.ndarray:
        """(H, W) uint8 class-id map of everything accumulated so far (0 = no data)."""
        covered = self._weight > 1e-3
        out = np.zeros((self.n_rows, self.n_cols), dtype=np.uint8)
        pred_channel = (self._prob_sum / np.maximum(self._weight, 1e-9)).argmax(axis=0)
        id_lookup = np.array(self.class_ids, dtype=np.uint8)
        out[covered] = id_lookup[pred_channel[covered]]
        return out


def build_class_map(
    records, model, stats, bands, class_ids, nodata, band_modes, device
) -> tuple[np.ndarray, object, object]:
    """(uint8 class-id map, North-up transform, crs) from mask-weighted blended probs."""
    acc = MapAccumulator(records, class_ids, nodata)
    for r in records:
        inputs = apply_stats(r.features, r.polygon, bands, stats, nodata, band_modes)
        acc.add(r, predict_probs(model, inputs, device))
    return acc.class_map(), acc.transform, acc.crs


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
    parser.add_argument("--device", default=None, help="Override train.device (cpu/mps/cuda).")
    args = parser.parse_args(argv)

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)
    if args.device:
        cfg.train.device = args.device
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
    stats = resolve_polygon_stats(cfg, polygon, records, stats, band_modes)

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
