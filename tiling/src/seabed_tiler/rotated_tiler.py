# tiling/src/seabed_tiler/rotated_tiler.py
"""Rotation-aware tiling loop.

Mirrors tiler.run_tiling() but: (1) derives tile grid from the annotation polygon
footprint MBR rather than the axis-aligned extent, and (2) extracts each tile via
affine-rotated warping (rasterio.warp.reproject) rather than direct array slicing.

Two entry points:
- run_rotated_tiling():   the base rotated grid -> outputs/<name>/<run_tag>_rot/
- run_augmented_tiling(): deterministic augmentation passes (theta jitter + origin
  shifts from cfg.augmentation) -> outputs/<name>/<run_tag>_rotaug/. Each pass is a
  genuine re-warp of the master grid, not a transform of already-written tiles.

Neither output directory ever collides with the standard axis-aligned output.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling

from .config import Config
from .io_utils import feature_profile, label_profile, tile_id
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


def _augmented_out_dir(cfg: Config) -> Path:
    return cfg.base_dir / cfg.output.dir / cfg.name / (cfg.run_tag + "_rotaug")


def _collect_label_shapefiles(cfg: Config) -> list[Path]:
    """Return all annotation shapefile paths declared in cfg.labels."""
    if cfg.labels.kind == "shapefile":
        return [cfg.labels_path]
    paths = []
    for file_list in (cfg.labels.class_files or {}).values():
        for fname in file_list:
            paths.append(cfg.src_path / fname)
    return paths


def _tile_center_utm(win: RotatedTileWindow, tile_size_m: float) -> tuple[float, float]:
    """UTM coords of the tile center (used for spatial train/val/test splits)."""
    c, s = math.cos(win.theta), math.sin(win.theta)
    u_c = win.u_origin + tile_size_m / 2.0
    v_c = win.v_origin - tile_size_m / 2.0
    return u_c * c - v_c * s, u_c * s + v_c * c


def _footprint_mbr(cfg: Config):
    """Compute (theta, mbr_corners) of the annotation footprint MBR."""
    shapefiles = _collect_label_shapefiles(cfg)
    footprint = compute_label_footprint(shapefiles)
    _, theta, mbr_corners = minimum_bounding_rect(footprint)
    return theta, mbr_corners


def _process_windows(
    cfg: Config,
    grid: dict,
    windows: list[RotatedTileWindow],
    out_dir: Path,
    id_prefix: str,
    features_resampling: Resampling,
    extra_meta: dict | None = None,
) -> list[dict]:
    """Extract, filter and write tiles for the given windows; return manifest rows."""
    transform = grid["transform"]
    crs = grid["crs"]
    nodata = grid["nodata"]
    label_nodata = cfg.output.label_nodata

    res = cfg.target_resolution_m
    tpx = int(round(cfg.tile_size_m / res))

    band_names = list(grid["features"].keys())
    feat_stack = np.stack([grid["features"][b] for b in band_names], axis=0)  # (B, H, W)
    label_stack = grid["label"][np.newaxis, :, :].astype("float32")  # (1, H, W)

    feat_dir = out_dir / "tiles" / "features"
    lab_dir = out_dir / "tiles" / "labels"
    feat_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    inv_classes = {v: k for k, v in cfg.labels.classes.items()}
    n_classes = max(cfg.labels.classes.values()) + 1

    rows: list[dict] = []
    for win in windows:
        feat_tile, tile_transform = extract_rotated_tile(
            feat_stack, transform, crs, win, res, nodata, tpx,
            resampling=features_resampling,
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

        tid = tile_id(id_prefix, win.row, win.col)
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

        center_x, center_y = _tile_center_utm(win, cfg.tile_size_m)
        counts = np.bincount(lab_tile.ravel(), minlength=n_classes)
        row_meta = {
            "tile_id": tid,
            "row": win.row,
            "col": win.col,
            "theta_deg": round(float(np.degrees(win.theta)), 4),
            "u_origin": win.u_origin,
            "v_origin": win.v_origin,
            "center_x": round(center_x, 3),
            "center_y": round(center_y, 3),
            "valid_frac": round(valid_frac, 4),
            "features_path": str(fpath.relative_to(cfg.base_dir)),
            "label_path": str(lpath.relative_to(cfg.base_dir)),
        }
        if extra_meta:
            row_meta.update(extra_meta)
        for class_id in range(n_classes):
            label = "background" if class_id == label_nodata else inv_classes.get(
                class_id, f"class{class_id}"
            )
            row_meta[f"{label}_px"] = int(counts[class_id])
        rows.append(row_meta)

    return rows


def run_rotated_tiling(cfg: Config, grid: dict) -> tuple[list[dict], list[RotatedTileWindow]]:
    """Write rotation-aware tiles; return (manifest_rows, all_candidate_windows)."""
    theta, mbr_corners = _footprint_mbr(cfg)
    windows = build_rotated_windows(mbr_corners, theta, cfg.tile_size_m, cfg.stride_m)

    out_dir = _rotated_out_dir(cfg)
    rows = _process_windows(
        cfg, grid, windows, out_dir,
        id_prefix=cfg.name,
        features_resampling=Resampling.nearest,
    )
    logger.info("rotated tiling complete: %d tiles written -> %s", len(rows), out_dir)
    return rows, windows


def run_augmented_tiling(cfg: Config, grid: dict) -> tuple[list[dict], list[RotatedTileWindow]]:
    """Run every augmentation pass from cfg.augmentation; return (rows, windows).

    Each pass re-extracts the rotated grid with the MBR angle shifted by
    theta_offset_deg and the grid origin shifted into the MBR by
    (u_shift_frac, v_shift_frac) fractions of the stride. Features are warped with
    bilinear resampling (continuous physical fields; source nodata is masked by
    rasterio so it does not bleed), labels always with nearest (class ids must
    never interpolate).
    """
    if not cfg.augmentation.enabled or not cfg.augmentation.passes:
        raise ValueError(
            "augmentation is not enabled: set augmentation.enabled=true and define "
            "augmentation.passes in the config (or pass --augment with a valid config)"
        )

    theta, mbr_corners = _footprint_mbr(cfg)
    out_dir = _augmented_out_dir(cfg)

    all_rows: list[dict] = []
    all_windows: list[RotatedTileWindow] = []
    for pass_idx, aug in enumerate(cfg.augmentation.passes, start=1):
        theta_pass = theta + math.radians(aug.theta_offset_deg)
        # v axis points up in the rotated frame and the grid walks downward from
        # v_max, so a positive v shift moves the grid INTO the MBR via -v_offset.
        windows = build_rotated_windows(
            mbr_corners, theta_pass, cfg.tile_size_m, cfg.stride_m,
            u_offset=aug.u_shift_frac * cfg.stride_m,
            v_offset=-aug.v_shift_frac * cfg.stride_m,
        )
        rows = _process_windows(
            cfg, grid, windows, out_dir,
            id_prefix=f"{cfg.name}_p{pass_idx:02d}",
            features_resampling=Resampling.bilinear,
            extra_meta={
                "aug_pass": pass_idx,
                "theta_offset_deg": aug.theta_offset_deg,
                "u_shift_frac": aug.u_shift_frac,
                "v_shift_frac": aug.v_shift_frac,
            },
        )
        logger.info(
            "augmentation pass %d/%d (dtheta=%.1f deg, du=%.2f, dv=%.2f): %d tiles",
            pass_idx, len(cfg.augmentation.passes),
            aug.theta_offset_deg, aug.u_shift_frac, aug.v_shift_frac, len(rows),
        )
        all_rows.extend(rows)
        all_windows.extend(windows)

    logger.info("augmented tiling complete: %d tiles written -> %s", len(all_rows), out_dir)
    return all_rows, all_windows
