"""Small helpers for tile naming and rasterio GeoTIFF profiles."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def clean_run_dir(out_dir: Path) -> None:
    """Delete a run output directory before regenerating it.

    Runs are deterministic and fully regenerated, but the tiler only overwrites
    files it writes this run: when a code or config change shrinks the tile set,
    tiles from the previous run would survive on disk -- and to_jpg converts every
    tif it finds, so stale tiles keep reappearing in the JPEG previews even though
    the manifest no longer lists them.
    """
    if out_dir.exists():
        logger.info("removing previous run outputs: %s", out_dir)
        shutil.rmtree(out_dir)


def tile_id(name: str, row: int, col: int) -> str:
    """Stable, sortable tile identifier, e.g. ``polygon1_r003_c007``."""
    return f"{name}_r{row:03d}_c{col:03d}"


def feature_profile(height, width, count, transform, crs, nodata, compress):
    """rasterio profile for a multiband float32 feature tile."""
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": count,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": compress,
    }


def label_profile(height, width, transform, crs, nodata, compress):
    """rasterio profile for a single-band uint8 label tile."""
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
        "compress": compress,
    }
