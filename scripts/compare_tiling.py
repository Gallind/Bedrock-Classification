"""Visual comparison of axis-aligned vs rotation-aware tiling for a polygon.

Draws both tile grids on the same stitched backscatter background so you can
see exactly where and how the two approaches differ.

Run:
    python scripts/compare_tiling.py --polygon polygon1
    python scripts/compare_tiling.py --polygon polygon5
    python scripts/compare_tiling.py --all
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import rasterio
import rasterio.plot
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RUN_TAG = "t128m_o50pct_r1m"
POLYGONS = ["polygon1", "polygon3", "polygon4", "polygon5"]

ORIG_COLOR = "#00CFFF"
ROT_COLOR  = "#FF6B35"


def _normalize(arr: np.ndarray, nodata: float) -> np.ndarray:
    valid = (arr != nodata) & ~np.isnan(arr)
    if valid.any():
        vmin, vmax = np.percentile(arr[valid], [2, 98])
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    out = np.clip((arr.astype("float32") - vmin) / (vmax - vmin), 0.0, 1.0)
    out[~valid] = np.nan
    return out


def compare_polygon(polygon: str, out_dir: Path) -> Path:
    base     = Path("outputs") / polygon
    orig_dir = base / RUN_TAG
    rot_dir  = base / f"{RUN_TAG}_rot"

    if not orig_dir.exists():
        raise FileNotFoundError(f"original output missing: {orig_dir}")
    if not rot_dir.exists():
        raise FileNotFoundError(f"rotated output missing: {rot_dir}")

    # Load stitched backscatter as background (first band of original features.tif)
    feat_tif = orig_dir / "stitched" / "features.tif"
    with rasterio.open(feat_tif) as ds:
        crs_utm = ds.crs.to_string()
        band_names = [d or f"band{i+1}" for i, d in enumerate(ds.descriptions)]
        # prefer backscatter; fall back to first band
        bi = (band_names.index("backscatter") + 1) if "backscatter" in band_names else 1
        arr = ds.read(bi).astype("float32")
        nodata = float(ds.nodata) if ds.nodata is not None else -9999.0
        transform = ds.transform
        bounds = ds.bounds
        height, width = arr.shape

    bg = _normalize(arr, nodata)

    # Load original tile footprints from manifest.csv
    df_orig = pd.read_csv(orig_dir / "manifest.csv")
    geoms_orig = [box(r.xmin, r.ymin, r.xmax, r.ymax) for _, r in df_orig.iterrows()]
    gdf_orig = gpd.GeoDataFrame(df_orig, geometry=geoms_orig, crs=crs_utm)

    # Load rotated tile footprints from manifest.geojson (WGS84) -> reproject to UTM
    gdf_rot = gpd.read_file(rot_dir / "manifest.geojson").to_crs(crs_utm)

    theta = gdf_rot["theta_deg"].iloc[0] if "theta_deg" in gdf_rot.columns else 0.0

    # Compute extent for imshow: [left, right, bottom, top] in UTM meters
    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)

    titles = [
        f"Original only  ({len(gdf_orig)} tiles)",
        f"Rotated only  ({len(gdf_rot)} tiles, theta={theta:.1f}°)",
        "Both overlaid",
    ]
    datasets = [
        (gdf_orig, None),
        (None, gdf_rot),
        (gdf_orig, gdf_rot),
    ]

    for ax, title, (go, gr) in zip(axes, titles, datasets):
        # Background
        ax.imshow(
            bg, cmap="gray", extent=extent, origin="upper",
            aspect="equal", interpolation="nearest",
            vmin=0.0, vmax=1.0,
        )
        if go is not None:
            go.boundary.plot(
                ax=ax, color=ORIG_COLOR, linewidth=0.8, alpha=0.85,
                label=f"Original ({len(go)})",
            )
        if gr is not None:
            gr.boundary.plot(
                ax=ax, color=ROT_COLOR, linewidth=1.1, alpha=0.9,
                label=f"Rotated ({len(gr)})",
            )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("UTM Easting (m)", fontsize=8)
        ax.set_ylabel("UTM Northing (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        if go is not None or gr is not None:
            ax.legend(fontsize=8, loc="upper left")

    orig_patch = mpatches.Patch(color=ORIG_COLOR, label=f"Original ({len(gdf_orig)} tiles)")
    rot_patch  = mpatches.Patch(color=ROT_COLOR,  label=f"Rotated  ({len(gdf_rot)} tiles, theta={theta:.1f}°)")
    fig.legend(handles=[orig_patch, rot_patch], loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(f"{polygon} — axis-aligned vs rotation-aware tiling", fontsize=14, fontweight="bold")

    out_path = out_dir / f"{polygon}_tiling_comparison.jpg"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved -> %s", out_path)
    return out_path


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Compare axis-aligned vs rotated tiling grids.")
    ap.add_argument("--polygon", help="e.g. polygon1")
    ap.add_argument("--all", action="store_true", help="Run for all polygons")
    ap.add_argument("--out-dir", default="outputs/comparisons",
                    help="Directory to write comparison images (default: outputs/comparisons)")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = POLYGONS if args.all else ([args.polygon] if args.polygon else [])
    if not targets:
        ap.error("specify --polygon NAME or --all")

    for poly in targets:
        try:
            compare_polygon(poly, out_dir)
        except FileNotFoundError as e:
            logger.warning("skipping %s: %s", poly, e)


if __name__ == "__main__":
    main()
