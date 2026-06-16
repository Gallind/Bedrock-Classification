"""Stitch the overlapping tiles back into the full image, to verify the split round-trips.

Each tile is painted into a master array at its georeferenced position. Overlapping tiles
carry identical values (they were cut from the same master grid), so the mosaic is seamless;
tiles that were filtered out simply leave gaps, which makes the kept-coverage visible.

Outputs a full GeoTIFF plus normalized JPEG previews (per feature band, and a colorized
label image) that you can open next to the original DataBase/polygon1 layers.

Run:
  PYTHONPATH=tiling/src .venv/bin/python -m seabed_tiler.stitch --tiles-dir outputs/polygon1
"""

from __future__ import annotations

import logging
import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.transform import from_origin

from .logging_utils import add_file_handler, setup_logging
from .viz import colorize, label_to_rgb, normalize_band, resolve_styles, write_prj, write_worldfile

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)


def _gather_grid(paths, manifest_path=None):
    """Union extent + resolution/CRS for the tiles -> master grid definition.

    Resolution and CRS come from one tile; the union extent comes from the manifest when
    available (avoids opening every tile just to read bounds), else from a full scan.
    """
    with rasterio.open(paths[0]) as ds:
        res = ds.res[0]
        crs = ds.crs

    if manifest_path is not None and manifest_path.exists():
        df = pd.read_csv(manifest_path)
        xmin, ymin = float(df.xmin.min()), float(df.ymin.min())
        xmax, ymax = float(df.xmax.max()), float(df.ymax.max())
    else:
        xmin = ymin = math.inf
        xmax = ymax = -math.inf
        for p in paths:
            with rasterio.open(p) as ds:
                b = ds.bounds
            xmin, ymin = min(xmin, b.left), min(ymin, b.bottom)
            xmax, ymax = max(xmax, b.right), max(ymax, b.top)

    n_cols = int(round((xmax - xmin) / res))
    n_rows = int(round((ymax - ymin) / res))
    transform = from_origin(xmin, ymax, res, res)
    return xmin, ymax, res, n_rows, n_cols, transform, crs


def stitch_features(tiles_dir: Path, out_dir: Path, styles: dict):
    paths = sorted((tiles_dir / "tiles" / "features").glob("*.tif"))
    if not paths:
        logger.info("  no feature tiles found")
        return
    xmin, ymax, res, n_rows, n_cols, transform, crs = _gather_grid(
        paths, tiles_dir / "manifest.csv"
    )
    with rasterio.open(paths[0]) as ds:
        n_bands = ds.count
        band_names = [d or f"band{i+1}" for i, d in enumerate(ds.descriptions)]
        nodata = ds.nodata

    master = np.full((n_bands, n_rows, n_cols), nodata, dtype="float32")
    for p in paths:
        with rasterio.open(p) as ds:
            data = ds.read()
            b = ds.bounds
        col = int(round((b.left - xmin) / res))
        row = int(round((ymax - b.top) / res))
        h, w = data.shape[1], data.shape[2]
        master[:, row : row + h, col : col + w] = data

    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / "features.tif"
    profile = {
        "driver": "GTiff", "height": n_rows, "width": n_cols, "count": n_bands,
        "dtype": "float32", "crs": crs, "transform": transform, "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(master)
        for i, bn in enumerate(band_names, start=1):
            dst.set_band_description(i, bn)

    for bi, bn in enumerate(band_names):
        style = styles.get(bn)
        if style:
            img = colorize(
                master[bi], nodata, style["cmap"], hillshade=style.get("hillshade", False),
                dx=res, vert_exag=style.get("vert_exag", 5.0),
            )
        else:
            img, _ = normalize_band(master[bi], nodata)
        jp = out_dir / f"features_{bn}.jpg"
        Image.fromarray(img).save(jp, quality=90)
        write_worldfile(jp, transform)
        write_prj(jp, crs)

    logger.info(f"  stitched {len(paths)} tiles -> {tif} ({n_rows}x{n_cols}px) + {n_bands} JPEG previews")


def stitch_labels(tiles_dir: Path, out_dir: Path):
    paths = sorted((tiles_dir / "tiles" / "labels").glob("*.tif"))
    if not paths:
        logger.info("  no label tiles found")
        return
    xmin, ymax, res, n_rows, n_cols, transform, crs = _gather_grid(
        paths, tiles_dir / "manifest.csv"
    )
    with rasterio.open(paths[0]) as ds:
        nodata = ds.nodata if ds.nodata is not None else 0

    master = np.full((n_rows, n_cols), nodata, dtype="uint8")
    for p in paths:
        with rasterio.open(p) as ds:
            data = ds.read(1)
            b = ds.bounds
        col = int(round((b.left - xmin) / res))
        row = int(round((ymax - b.top) / res))
        h, w = data.shape
        master[row : row + h, col : col + w] = data

    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / "labels.tif"
    profile = {
        "driver": "GTiff", "height": n_rows, "width": n_cols, "count": 1,
        "dtype": "uint8", "crs": crs, "transform": transform, "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(master, 1)

    jp = out_dir / "labels.jpg"
    Image.fromarray(label_to_rgb(master)).save(jp, quality=90)
    write_worldfile(jp, transform)
    write_prj(jp, crs)
    logger.info(f"  stitched {len(paths)} tiles -> {tif} ({n_rows}x{n_cols}px) + colorized JPEG")


def _rotated_extent(paths) -> tuple[float, float, float, float, float]:
    """Return (xmin, ymin, xmax, ymax, res) as the axis-aligned envelope of all rotated tiles."""
    xmin = ymin = math.inf
    xmax = ymax = -math.inf
    res = None
    for p in paths:
        with rasterio.open(p) as ds:
            b = ds.bounds
            xmin = min(xmin, b.left)
            ymin = min(ymin, b.bottom)
            xmax = max(xmax, b.right)
            ymax = max(ymax, b.top)
            if res is None:
                res = math.hypot(ds.transform.a, ds.transform.d)
    return xmin, ymin, xmax, ymax, res


def stitch_rotated_features(rot_tiles_dir: Path, out_dir: Path, styles: dict) -> None:
    """Stitch rotated feature tiles into a North-up mosaic by reprojecting each tile."""
    from rasterio.warp import reproject, Resampling as _Resampling
    from rasterio.transform import from_origin
    paths = sorted((rot_tiles_dir / "tiles" / "features").glob("*.tif"))
    if not paths:
        logger.info("  no rotated feature tiles found")
        return

    with rasterio.open(paths[0]) as ds:
        crs = ds.crs
        nodata = ds.nodata
        n_bands = ds.count
        band_names = [d or f"band{i+1}" for i, d in enumerate(ds.descriptions)]

    xmin, ymin, xmax, ymax, res = _rotated_extent(paths)
    n_cols = int(round((xmax - xmin) / res))
    n_rows = int(round((ymax - ymin) / res))
    dst_transform = from_origin(xmin, ymax, res, res)

    master = np.full((n_bands, n_rows, n_cols), nodata, dtype="float32")
    for p in paths:
        with rasterio.open(p) as ds:
            for bi in range(n_bands):
                dst_band = np.full((n_rows, n_cols), nodata, dtype="float32")
                reproject(
                    source=rasterio.band(ds, bi + 1),
                    destination=dst_band,
                    src_transform=ds.transform,
                    src_crs=ds.crs,
                    dst_transform=dst_transform,
                    dst_crs=crs,
                    resampling=_Resampling.nearest,
                    src_nodata=nodata,
                    dst_nodata=nodata,
                )
                valid = dst_band != nodata
                master[bi][valid] = dst_band[valid]

    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / "features.tif"
    profile = {
        "driver": "GTiff", "height": n_rows, "width": n_cols, "count": n_bands,
        "dtype": "float32", "crs": crs, "transform": dst_transform,
        "nodata": nodata, "compress": "deflate",
    }
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(master)
        for i, bn in enumerate(band_names, start=1):
            dst.set_band_description(i, bn)

    for bi, bn in enumerate(band_names):
        style = styles.get(bn)
        if style:
            img = colorize(
                master[bi], nodata, style["cmap"], hillshade=style.get("hillshade", False),
                dx=res, vert_exag=style.get("vert_exag", 5.0),
            )
        else:
            img, _ = normalize_band(master[bi], nodata)
        jp = out_dir / f"features_{bn}.jpg"
        Image.fromarray(img).save(jp, quality=90)
        write_worldfile(jp, dst_transform)
        write_prj(jp, crs)

    logger.info(f"  stitched {len(paths)} rotated feature tiles -> {tif} ({n_rows}x{n_cols}px) + {n_bands} JPEG previews")


def stitch_rotated_labels(rot_tiles_dir: Path, out_dir: Path) -> None:
    """Stitch rotated label tiles into a North-up mosaic by reprojecting each tile."""
    from rasterio.warp import reproject, Resampling as _Resampling
    from rasterio.transform import from_origin
    paths = sorted((rot_tiles_dir / "tiles" / "labels").glob("*.tif"))
    if not paths:
        logger.info("  no rotated label tiles found")
        return

    with rasterio.open(paths[0]) as ds:
        crs = ds.crs
        nodata = int(ds.nodata) if ds.nodata is not None else 0

    xmin, ymin, xmax, ymax, res = _rotated_extent(paths)
    n_cols = int(round((xmax - xmin) / res))
    n_rows = int(round((ymax - ymin) / res))
    dst_transform = from_origin(xmin, ymax, res, res)

    master = np.full((n_rows, n_cols), nodata, dtype="uint8")
    for p in paths:
        with rasterio.open(p) as ds:
            dst_band = np.full((n_rows, n_cols), nodata, dtype="float32")
            reproject(
                source=rasterio.band(ds, 1),
                destination=dst_band,
                src_transform=ds.transform,
                src_crs=ds.crs,
                dst_transform=dst_transform,
                dst_crs=crs,
                resampling=_Resampling.nearest,
                src_nodata=float(nodata),
                dst_nodata=float(nodata),
            )
            valid = dst_band != nodata
            master[valid] = dst_band[valid].astype("uint8")

    out_dir.mkdir(parents=True, exist_ok=True)
    tif = out_dir / "labels.tif"
    profile = {
        "driver": "GTiff", "height": n_rows, "width": n_cols, "count": 1,
        "dtype": "uint8", "crs": crs, "transform": dst_transform,
        "nodata": nodata, "compress": "deflate",
    }
    with rasterio.open(tif, "w", **profile) as dst:
        dst.write(master, 1)

    jp = out_dir / "labels.jpg"
    Image.fromarray(label_to_rgb(master)).save(jp, quality=90)
    write_worldfile(jp, dst_transform)
    write_prj(jp, crs)
    logger.info(f"  stitched {len(paths)} rotated label tiles -> {tif} ({n_rows}x{n_cols}px) + colorized JPEG")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Stitch tiles back into the full image for QA.")
    ap.add_argument("--tiles-dir", default=None,
                    help="Run folder containing tiles/features and tiles/labels "
                         "(default: derived from --config).")
    ap.add_argument("--out", default=None, help="Output dir (default: <tiles-dir>/stitched).")
    ap.add_argument("--what", choices=["features", "labels", "both"], default="both")
    ap.add_argument("--config", default="tiling/config/polygon1.yaml",
                    help="Polygon config for per-band colormaps (falls back to built-in defaults).")
    ap.add_argument("--gray", action="store_true",
                    help="Force grayscale for all feature bands (ignore colormaps).")
    args = ap.parse_args(argv)

    if args.tiles_dir:
        tiles_dir = Path(args.tiles_dir)
    else:
        from .config import load_config
        tiles_dir = load_config(args.config).out_dir
    out_dir = Path(args.out) if args.out else tiles_dir / "stitched"
    setup_logging()
    add_file_handler(tiles_dir / "stitch.log")
    styles = {} if args.gray else resolve_styles(args.config)

    logger.info(f"[+] stitch tiles ({args.what}) from {tiles_dir}")
    if args.what in ("features", "both"):
        stitch_features(tiles_dir, out_dir, styles)
    if args.what in ("labels", "both"):
        stitch_labels(tiles_dir, out_dir)
    logger.info(f"[+] done -> {out_dir}")


if __name__ == "__main__":
    main()
