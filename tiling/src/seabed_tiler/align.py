"""Build the master 0.5 m grid and resample every feature layer onto it.

All layers end up on one common grid (extent + resolution + CRS) so they can be stacked
band-for-band and tiled together. Raster layers are warped; XYZ layers are snapped.
"""

from __future__ import annotations

import math

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from .config import Config, LayerConfig
from .xyz import grid_xyz, load_xyz_df

_RESAMPLING = {
    "nearest": Resampling.nearest,
    "average": Resampling.average,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "mode": Resampling.mode,
}


def _resampling(name: str) -> Resampling:
    try:
        return _RESAMPLING[name]
    except KeyError:
        raise ValueError(
            f"Unknown resampling '{name}'. Options: {sorted(_RESAMPLING)}"
        )


def _layer_bounds(cfg: Config, layer: LayerConfig, xyz_cache: dict) -> tuple:
    """(xmin, ymin, xmax, ymax) for a layer, caching loaded XYZ DataFrames."""
    path = cfg.layer_path(layer)
    if layer.kind == "raster_jpg":
        with rasterio.open(path) as src:
            b = src.bounds
        return (b.left, b.bottom, b.right, b.top)
    df = load_xyz_df(path)
    xyz_cache[layer.name] = df
    return (float(df.x.min()), float(df.y.min()), float(df.x.max()), float(df.y.max()))


def _reproject_raster(
    cfg: Config, layer: LayerConfig, crs, transform, n_rows, n_cols, nodata
) -> np.ndarray:
    """Read a JPEG render, optionally convert RGB->gray, warp onto the master grid."""
    with rasterio.open(cfg.layer_path(layer)) as src:
        data = src.read()  # (bands, H, W)
        src_transform = src.transform
        src_crs = src.crs or CRS.from_string(cfg.crs)

    if layer.to_gray and data.shape[0] >= 3:
        source = data[:3].mean(axis=0).astype("float32")
    else:
        source = data[0].astype("float32")

    dst = np.full((n_rows, n_cols), nodata, dtype="float32")
    reproject(
        source=source,
        destination=dst,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=crs,
        resampling=_resampling(layer.resampling),
        dst_nodata=nodata,
    )
    return dst


def build_grid_and_features(cfg: Config) -> dict:
    """Resolve the master grid and produce one aligned float32 array per feature layer.

    Returns a dict with keys: ``transform``, ``crs``, ``shape`` (rows, cols),
    ``extent`` (xmin, ymin, xmax, ymax), ``features`` (ordered by ``band_order``),
    and ``nodata``.
    """
    crs = CRS.from_string(cfg.crs)
    res = cfg.target_resolution_m
    nodata = cfg.output.feature_nodata

    xyz_cache: dict = {}
    bounds = [_layer_bounds(cfg, layer, xyz_cache) for layer in cfg.layers]

    if cfg.extent == "auto":
        # Intersection of all layers — every output pixel is backed by every layer's extent.
        xmin = max(b[0] for b in bounds)
        ymin = max(b[1] for b in bounds)
        xmax = min(b[2] for b in bounds)
        ymax = min(b[3] for b in bounds)
    else:
        xmin, ymin, xmax, ymax = cfg.extent

    # Snap the grid outward to a round origin so tiles are reproducible across runs.
    snap = cfg.origin_snap_m
    xmin = math.floor(xmin / snap) * snap
    ymin = math.floor(ymin / snap) * snap
    xmax = math.ceil(xmax / snap) * snap
    ymax = math.ceil(ymax / snap) * snap

    n_cols = int(round((xmax - xmin) / res))
    n_rows = int(round((ymax - ymin) / res))
    transform = from_origin(xmin, ymax, res, res)

    features: dict[str, np.ndarray] = {}
    for layer in cfg.layers:
        if layer.kind == "raster_jpg":
            features[layer.name] = _reproject_raster(
                cfg, layer, crs, transform, n_rows, n_cols, nodata
            )
        else:
            df = xyz_cache[layer.name]
            features[layer.name] = grid_xyz(df, xmin, ymax, n_rows, n_cols, res, nodata)

    ordered = {name: features[name] for name in cfg.band_order}

    return {
        "transform": transform,
        "crs": crs,
        "shape": (n_rows, n_cols),
        "extent": (xmin, ymin, xmax, ymax),
        "features": ordered,
        "nodata": nodata,
    }
