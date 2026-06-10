"""Core loop: cut the aligned feature stack + label into overlapping GeoTIFF tiles.

Each window from grid.build_windows is sliced out of the master arrays, filtered on
coverage/label presence, and written as two co-registered GeoTIFFs (multiband features +
single-band labels), each carrying its own CRS and transform so geolocation is intact.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.windows import transform as window_transform

from .config import Config
from .grid import build_windows
from .io_utils import clean_run_dir, feature_profile, label_profile, tile_id


def run_tiling(cfg: Config, grid: dict) -> tuple[list[dict], list]:
    """Write tiles and return (manifest_rows, all_candidate_windows)."""
    transform = grid["transform"]
    crs = grid["crs"]
    n_rows, n_cols = grid["shape"]
    xmin, ymin, xmax, ymax = grid["extent"]
    nodata = grid["nodata"]
    label_arr = grid["label"]

    res = cfg.target_resolution_m
    tpx = int(round(cfg.tile_size_m / res))

    band_names = list(grid["features"].keys())
    stack = np.stack([grid["features"][b] for b in band_names], axis=0)  # (B, H, W)

    label_nodata = cfg.output.label_nodata
    inv_classes = {v: k for k, v in cfg.labels.classes.items()}
    n_classes = max(cfg.labels.classes.values()) + 1

    clean_run_dir(cfg.out_dir)
    feat_dir = cfg.out_dir / "tiles" / "features"
    lab_dir = cfg.out_dir / "tiles" / "labels"
    feat_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)

    windows = build_windows(
        (xmin, ymin, xmax, ymax), cfg.tile_size_m, cfg.stride_m, cfg.keep_partial_edge
    )

    rows: list[dict] = []
    for win in windows:
        col_off = int(round((win.xmin - xmin) / res))
        row_off = int(round((ymax - win.ymax) / res))
        if row_off < 0 or col_off < 0 or row_off >= n_rows or col_off >= n_cols:
            continue
        h = min(tpx, n_rows - row_off)
        w = min(tpx, n_cols - col_off)
        if not cfg.keep_partial_edge and (h < tpx or w < tpx):
            continue

        feat_tile = stack[:, row_off : row_off + h, col_off : col_off + w]
        lab_tile = label_arr[row_off : row_off + h, col_off : col_off + w]

        # A pixel is "valid" only where every feature band has real data.
        valid = np.all(feat_tile != nodata, axis=0) & ~np.any(np.isnan(feat_tile), axis=0)
        valid_frac = float(valid.mean()) if valid.size else 0.0
        has_label = bool(np.any(lab_tile != label_nodata))

        if valid_frac < cfg.filters.min_valid_frac:
            continue
        if cfg.filters.require_label and not has_label:
            continue

        rio_win = Window(col_off, row_off, w, h)
        tile_transform = window_transform(rio_win, transform)
        tid = tile_id(cfg.name, win.row, win.col)
        fpath = feat_dir / f"{tid}.tif"
        lpath = lab_dir / f"{tid}.tif"

        fprofile = feature_profile(
            h, w, len(band_names), tile_transform, crs, nodata, cfg.output.compress
        )
        with rasterio.open(fpath, "w", **fprofile) as dst:
            dst.write(feat_tile.astype("float32"))
            for i, bn in enumerate(band_names, start=1):
                dst.set_band_description(i, bn)

        lprofile = label_profile(
            h, w, tile_transform, crs, label_nodata, cfg.output.compress
        )
        with rasterio.open(lpath, "w", **lprofile) as dst:
            dst.write(lab_tile.astype("uint8"), 1)

        counts = np.bincount(lab_tile.ravel(), minlength=n_classes)
        row = {
            "tile_id": tid,
            "row": win.row,
            "col": win.col,
            "xmin": win.xmin,
            "ymin": win.ymin,
            "xmax": win.xmax,
            "ymax": win.ymax,
            "valid_frac": round(valid_frac, 4),
            "features_path": str(fpath.relative_to(cfg.base_dir)),
            "label_path": str(lpath.relative_to(cfg.base_dir)),
        }
        for class_id in range(n_classes):
            label = "background" if class_id == label_nodata else inv_classes.get(
                class_id, f"class{class_id}"
            )
            row[f"{label}_px"] = int(counts[class_id])
        rows.append(row)

    return rows, windows
