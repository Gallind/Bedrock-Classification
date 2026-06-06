# tiling/src/seabed_tiler/rotated_tiler.py
"""Rotation-aware tiling loop.

Mirrors tiler.run_tiling() but: (1) derives tile grid from the annotation polygon
footprint MBR rather than the axis-aligned extent, and (2) extracts each tile via
affine-rotated warping (rasterio.warp.reproject) rather than direct array slicing.

Output directory: outputs/<name>/<run_tag>_rot/ -- never collides with standard output.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from .config import Config
from .io_utils import feature_profile, label_profile, tile_id
from .manifest import write_rotated_manifest
from .rotation import (
    RotatedTileWindow,
    build_rotated_windows,
    compute_label_footprint,
    extract_rotated_tile,
    minimum_bounding_rect,
)

logger = logging.getLogger(__name__)


def _rotated_out_dir(cfg: Config) -> Path:
    return cfg.base_dir / cfg.output.dir / cfg.name / (cfg.run_tag + "_rot")


def _collect_label_shapefiles(cfg: Config) -> list[Path]:
    """Return all annotation shapefile paths declared in cfg.labels."""
    if cfg.labels.kind == "shapefile":
        return [cfg.labels_path]
    paths = []
    for file_list in (cfg.labels.class_files or {}).values():
        for fname in file_list:
            paths.append(cfg.src_path / fname)
    return paths


def run_rotated_tiling(cfg: Config, grid: dict) -> tuple[list[dict], list[RotatedTileWindow]]:
    """Write rotation-aware tiles; return (manifest_rows, all_candidate_windows)."""
    transform = grid["transform"]
    crs = grid["crs"]
    nodata = grid["nodata"]
    label_arr = grid["label"]
    label_nodata = cfg.output.label_nodata

    res = cfg.target_resolution_m
    tpx = int(round(cfg.tile_size_m / res))

    band_names = list(grid["features"].keys())
    feat_stack = np.stack([grid["features"][b] for b in band_names], axis=0)  # (B, H, W)
    label_stack = label_arr[np.newaxis, :, :].astype("float32")  # (1, H, W)

    shapefiles = _collect_label_shapefiles(cfg)
    footprint = compute_label_footprint(shapefiles)
    mbr, theta, mbr_corners = minimum_bounding_rect(footprint)
    windows = build_rotated_windows(mbr_corners, theta, cfg.tile_size_m, cfg.stride_m)

    out_dir = _rotated_out_dir(cfg)
    feat_dir = out_dir / "tiles" / "features"
    lab_dir = out_dir / "tiles" / "labels"
    feat_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    inv_classes = {v: k for k, v in cfg.labels.classes.items()}
    n_classes = max(cfg.labels.classes.values()) + 1
    theta_deg = round(float(np.degrees(theta)), 4)

    rows: list[dict] = []
    for win in windows:
        feat_tile, tile_transform = extract_rotated_tile(
            feat_stack, transform, crs, win, res, nodata, tpx,
            resampling=Resampling.nearest,
        )
        lab_tile, _ = extract_rotated_tile(
            label_stack, transform, crs, win, res, float(label_nodata), tpx,
            resampling=Resampling.nearest,
        )
        lab_tile = lab_tile[0].astype("uint8")

        valid = np.all(feat_tile != nodata, axis=0) & ~np.any(np.isnan(feat_tile), axis=0)
        valid_frac = float(valid.mean()) if valid.size else 0.0
        has_label = bool(np.any(lab_tile != label_nodata))

        if valid_frac < cfg.filters.min_valid_frac:
            continue
        if cfg.filters.require_label and not has_label:
            continue

        tid = tile_id(cfg.name, win.row, win.col)
        fpath = feat_dir / f"{tid}.tif"
        lpath = lab_dir / f"{tid}.tif"

        fprofile = feature_profile(
            tpx, tpx, len(band_names), tile_transform, crs, nodata, cfg.output.compress,
        )
        with rasterio.open(fpath, "w", **fprofile) as dst:
            dst.write(feat_tile.astype("float32"))
            for i, bn in enumerate(band_names, start=1):
                dst.set_band_description(i, bn)

        lprofile = label_profile(tpx, tpx, tile_transform, crs, label_nodata, cfg.output.compress)
        with rasterio.open(lpath, "w", **lprofile) as dst:
            dst.write(lab_tile, 1)

        counts = np.bincount(lab_tile.ravel(), minlength=n_classes)
        row_meta = {
            "tile_id": tid,
            "row": win.row,
            "col": win.col,
            "theta_deg": theta_deg,
            "u_origin": win.u_origin,
            "v_origin": win.v_origin,
            "valid_frac": round(valid_frac, 4),
            "features_path": str(fpath.relative_to(cfg.base_dir)),
            "label_path": str(lpath.relative_to(cfg.base_dir)),
        }
        for class_id in range(n_classes):
            label = "background" if class_id == label_nodata else inv_classes.get(
                class_id, f"class{class_id}"
            )
            row_meta[f"{label}_px"] = int(counts[class_id])
        rows.append(row_meta)

    logger.info("rotated tiling complete: %d tiles written -> %s", len(rows), out_dir)
    return rows, windows
