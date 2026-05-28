"""Convert the GeoTIFF tiles to viewable JPEGs (with .jgw/.prj sidecars).

Feature tiles are float32 with 3 bands; each band is rendered to its own grayscale JPEG
using a consistent intensity range (estimated once from a sample of tiles so brightness is
comparable across tiles). Label tiles are colorized by class.

Run:
  PYTHONPATH=tiling/src .venv/bin/python -m seabed_tiler.to_jpg --tiles-dir outputs/polygon1
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

from .viz import colorize, label_to_rgb, normalize_band, resolve_styles, write_prj, write_worldfile


def _save_jpg(array, path: Path, quality: int = 92) -> None:
    Image.fromarray(array).save(path, quality=quality)


def _band_ranges(paths, n_bands, nodata, sample=400, per_band_px=2000):
    """Estimate a (vmin, vmax) per band from a sample of tiles for consistent scaling."""
    idx = np.unique(np.linspace(0, len(paths) - 1, min(sample, len(paths))).astype(int))
    buckets: list[list] = [[] for _ in range(n_bands)]
    for i in idx:
        with rasterio.open(paths[i]) as ds:
            data = ds.read()
        for b in range(n_bands):
            valid = data[b][(data[b] != nodata) & ~np.isnan(data[b])].ravel()
            if valid.size:
                if valid.size > per_band_px:
                    valid = np.random.default_rng(0).choice(valid, per_band_px, replace=False)
                buckets[b].append(valid)
    ranges = []
    for b in range(n_bands):
        if buckets[b]:
            ranges.append(tuple(np.percentile(np.concatenate(buckets[b]), [2, 98])))
        else:
            ranges.append((0.0, 1.0))
    return ranges


def convert_features(tiles_dir: Path, out_dir: Path, limit: int | None, worldfile: bool,
                     styles: dict):
    paths = sorted((tiles_dir / "tiles" / "features").glob("*.tif"))
    if limit:
        paths = paths[:limit]
    if not paths:
        print("  no feature tiles found")
        return

    with rasterio.open(paths[0]) as ds:
        n_bands = ds.count
        band_names = [d or f"band{i+1}" for i, d in enumerate(ds.descriptions)]
        nodata = ds.nodata
        dx = ds.res[0]

    ranges = _band_ranges(paths, n_bands, nodata)
    print(f"  feature bands {band_names}")
    print(f"  intensity ranges {[tuple(round(v, 2) for v in r) for r in ranges]}")
    for bn in band_names:
        style = styles.get(bn)
        print(f"    {bn:<12} -> {'cmap=' + style['cmap'] + (' +hillshade' if style.get('hillshade') else '') if style else 'grayscale'}")
        (out_dir / "features" / bn).mkdir(parents=True, exist_ok=True)

    for p in paths:
        with rasterio.open(p) as ds:
            data = ds.read()
            transform = ds.transform
            crs = ds.crs
        for bi, bn in enumerate(band_names):
            style = styles.get(bn)
            if style:
                img = colorize(
                    data[bi], nodata, style["cmap"], ranges[bi][0], ranges[bi][1],
                    hillshade=style.get("hillshade", False), dx=dx,
                    vert_exag=style.get("vert_exag", 5.0),
                )
            else:
                img, _ = normalize_band(data[bi], nodata, ranges[bi][0], ranges[bi][1])
            jp = out_dir / "features" / bn / f"{p.stem}.jpg"
            _save_jpg(img, jp)
            if worldfile:
                write_worldfile(jp, transform)
                write_prj(jp, crs)
    print(f"  wrote {len(paths)} tiles x {n_bands} bands -> {out_dir / 'features'}")


def convert_labels(tiles_dir: Path, out_dir: Path, limit: int | None, worldfile: bool):
    paths = sorted((tiles_dir / "tiles" / "labels").glob("*.tif"))
    if limit:
        paths = paths[:limit]
    if not paths:
        print("  no label tiles found")
        return

    (out_dir / "labels").mkdir(parents=True, exist_ok=True)
    for p in paths:
        with rasterio.open(p) as ds:
            arr = ds.read(1)
            transform = ds.transform
            crs = ds.crs
        jp = out_dir / "labels" / f"{p.stem}.jpg"
        _save_jpg(label_to_rgb(arr), jp)
        if worldfile:
            write_worldfile(jp, transform)
            write_prj(jp, crs)
    print(f"  wrote {len(paths)} label tiles -> {out_dir / 'labels'}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Convert GeoTIFF tiles to viewable JPEGs.")
    ap.add_argument("--tiles-dir", default=None,
                    help="Run folder containing tiles/features and tiles/labels "
                         "(default: derived from --config).")
    ap.add_argument("--out", default=None, help="Output dir (default: <tiles-dir>/jpg).")
    ap.add_argument("--what", choices=["features", "labels", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="Only convert the first N tiles.")
    ap.add_argument("--no-worldfile", action="store_true",
                    help="Skip writing .jgw/.prj sidecars.")
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
    out_dir = Path(args.out) if args.out else tiles_dir / "jpg"
    worldfile = not args.no_worldfile
    styles = {} if args.gray else resolve_styles(args.config)

    print(f"[+] tiles -> jpg  ({args.what})  from {tiles_dir}")
    if args.what in ("features", "both"):
        convert_features(tiles_dir, out_dir, args.limit, worldfile, styles)
    if args.what in ("labels", "both"):
        convert_labels(tiles_dir, out_dir, args.limit, worldfile)
    print(f"[+] done -> {out_dir}")


if __name__ == "__main__":
    main()
