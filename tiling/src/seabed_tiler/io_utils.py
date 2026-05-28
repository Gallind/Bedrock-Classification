"""Small helpers for tile naming and rasterio GeoTIFF profiles."""

from __future__ import annotations


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
