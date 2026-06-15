"""Prediction CLI: classified seabed map(s) for one polygon, one per model.

Per-pixel probabilities are blended across the 50%-overlapping _rot tiles via the
U-Net's MapAccumulator (mask-weighted average -> argmax), so map geometry matches the
U-Net maps. Optional categorical majority filter smooths salt-and-pepper.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from scipy.ndimage import uniform_filter

from seabed_tiler.viz import label_to_rgb, write_prj, write_worldfile
from seabed_unet.config import Config
from seabed_unet.data import load_run_records
from seabed_unet.logging_utils import add_file_handler, remove_handler, setup_logging
from seabed_unet.normalize import apply_stats, load_stats
from seabed_unet.predict import MapAccumulator, resolve_polygon_stats

from .config import ForestConfig, load_forest_config
from .model import load_model, predict_proba_channels

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def majority_filter(class_map: np.ndarray, size: int, class_ids: list[int]) -> np.ndarray:
    """NxN categorical majority filter; preserves nodata (0) pixels."""
    if size <= 1:
        return class_map
    counts = np.stack([
        uniform_filter((class_map == cid).astype(np.float64), size=size, mode="constant")
        for cid in class_ids
    ])
    winner = np.array(class_ids, dtype=np.uint8)[counts.argmax(axis=0)]
    winner[class_map == 0] = 0
    return winner


def _proba_field(estimator, inputs: np.ndarray, num_classes: int) -> np.ndarray:
    """(C, H, W) per-pixel probability field from normalized (B, H, W) features."""
    b, h, w = inputs.shape
    flat = inputs.reshape(b, h * w).T
    proba = predict_proba_channels(estimator, flat, num_classes)   # (H*W, C)
    return proba.T.reshape(num_classes, h, w)


def _write_map(class_map, transform, crs, out_dir: Path, polygon: str, kind: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tif_path = out_dir / f"{polygon}_pred_{kind}.tif"
    profile = {
        "driver": "GTiff", "height": class_map.shape[0], "width": class_map.shape[1],
        "count": 1, "dtype": "uint8", "crs": crs, "transform": transform,
        "nodata": 0, "compress": "deflate",
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(class_map, 1)
    jpg_path = out_dir / f"{polygon}_pred_{kind}.jpg"
    Image.fromarray(label_to_rgb(class_map)).save(jpg_path, quality=90)
    write_worldfile(jpg_path, transform)
    write_prj(jpg_path, crs)
    return tif_path


def predict_polygon_map(cfg: Config, forest: ForestConfig, polygon: str) -> list[Path]:
    """Write one GeoTIFF+JPEG map per model for ``polygon``; return the TIFF paths."""
    setup_logging()
    run_dir = cfg.run_dir
    handler = add_file_handler(run_dir / "forest_predict.log")
    try:
        band_modes = cfg.normalization.modes_for(cfg.bands)
        records = load_run_records(cfg.rot_dir(polygon), polygon, cfg.bands, cfg.base_dir, augmented=False)
        stats = load_stats(run_dir / "normalization_stats.json")
        stats = resolve_polygon_stats(cfg, polygon, records, stats, band_modes)
        logger.info(f"[+] {cfg.name}: mapping {polygon} from {len(records)} base tiles")

        out_paths: list[Path] = []
        for kind in forest.models:
            est = load_model(run_dir / f"model_{kind}.joblib")
            acc = MapAccumulator(records, cfg.class_ids, cfg.feature_nodata)
            for r in records:
                inputs = apply_stats(r.features, r.polygon, cfg.bands, stats, cfg.feature_nodata, band_modes)
                acc.add(r, _proba_field(est, inputs, cfg.num_classes))
            class_map = acc.class_map()
            if forest.majority_filter_size > 1:
                class_map = majority_filter(class_map, forest.majority_filter_size, cfg.class_ids)
            tif = _write_map(class_map, acc.transform, acc.crs, run_dir / "maps", polygon, kind)
            out_paths.append(tif)
            logger.info(f"    {kind}: {int((class_map > 0).sum()):,} classified px -> {tif.name}")
        return out_paths
    finally:
        remove_handler(handler)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Predict classified seabed map(s).")
    parser.add_argument("--config", required=True, help="Forest experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--polygon", default=None, help="Polygon to map (default: first split polygon).")
    args = parser.parse_args(argv)
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg, forest = load_forest_config(args.config, base_dir=base)
    polygon = args.polygon or (cfg.split.test[0] if cfg.split.test else cfg.split.polygons[0])
    predict_polygon_map(cfg, forest, polygon)


if __name__ == "__main__":
    main()
