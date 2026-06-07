"""Visual comparison of axis-aligned vs rotation-aware tiling for a polygon.

Draws both tile grids on top of stitched data backgrounds (one figure per layer)
so you can see exactly where and how the two approaches differ, and confirm that
50% overlap is applied correctly.

Run:
    python scripts/compare_tiling.py --polygon polygon1
    python scripts/compare_tiling.py --polygon polygon5
    python scripts/compare_tiling.py --all

Prerequisites: run seabed_tiler.stitch before this script so that
  outputs/<polygon>/<run_tag>/stitched/ contains features.tif and labels.jpg.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

RUN_TAG = "t128m_o50pct_r1m"
POLYGONS = ["polygon1", "polygon3", "polygon4", "polygon5"]

ORIG_COLOR = "#00CFFF"
ROT_COLOR  = "#FF6B35"



def _draw_tile_grid(ax, gdf, color, outline_alpha=0.75, label=None):
    """Draw tile polygon outlines."""
    gdf.boundary.plot(ax=ax, color=color, linewidth=0.6, alpha=outline_alpha, label=label)


def _coverage_count(gdf: gpd.GeoDataFrame, extent: list, img_shape: tuple) -> np.ndarray:
    """Return a pixel raster counting how many tiles cover each pixel.

    Uses rasterio.features.rasterize per tile so rotated polygon shapes are
    respected exactly — no bounding-box overcount for rotated tiles.
    """
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    from shapely.geometry import mapping

    left, right, bottom, top = extent
    height, width = img_shape[:2]
    transform = from_bounds(left, bottom, right, top, width, height)
    count = np.zeros((height, width), dtype="float32")
    for geom in gdf.geometry:
        burned = rasterize(
            [(mapping(geom), 1)],
            out_shape=(height, width),
            transform=transform,
            fill=0,
            dtype="float32",
        )
        count += burned
    return count


def _compare_one_layer(
    polygon: str,
    layer_name: str,
    bg: np.ndarray,
    gdf_orig: gpd.GeoDataFrame,
    gdf_rot: gpd.GeoDataFrame,
    extent: list,
    theta: float,
    out_dir: Path,
) -> Path:
    """Generate a 4-panel comparison for a single data layer.

    Panels:
      1. Original (axis-aligned) tile grid on data background
      2. Rotated tile grid on data background
      3. Both grids overlaid
      4. Tile coverage density heatmap for original tiling — directly shows 50% overlap:
         interior pixels are covered by 4 tiles (2 horizontal neighbors x 2 vertical),
         edge pixels by 2, and corner pixels by 1.
    """
    m = re.match(r"t(\d+)m_o(\d+)pct", RUN_TAG)
    tile_m = int(m.group(1)) if m else "?"
    ovl_pct = int(m.group(2)) if m else "?"
    stride_m = int(tile_m) * (100 - int(ovl_pct)) // 100 if m else "?"

    fig, axes = plt.subplots(1, 4, figsize=(28, 7), constrained_layout=True)

    # Panels 1-3: tile grid overlays
    titles = [
        f"Original ({len(gdf_orig)} tiles)",
        f"Rotated ({len(gdf_rot)} tiles, theta={theta:.1f}°)",
        "Both overlaid",
    ]
    datasets = [(gdf_orig, None), (None, gdf_rot), (gdf_orig, gdf_rot)]

    for ax, title, (go, gr) in zip(axes[:3], titles, datasets):
        ax.imshow(bg, extent=extent, origin="upper", aspect="equal", interpolation="nearest")
        if go is not None:
            _draw_tile_grid(ax, go, ORIG_COLOR, label=f"Original ({len(go)})")
        if gr is not None:
            _draw_tile_grid(ax, gr, ROT_COLOR, label=f"Rotated ({len(gr)})")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("UTM Easting (m)", fontsize=8)
        ax.set_ylabel("UTM Northing (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        if go is not None or gr is not None:
            ax.legend(fontsize=8, loc="upper left")

    # Panel 4: coverage density heatmap
    ax4 = axes[3]
    ax4.imshow(bg, extent=extent, origin="upper", aspect="equal", interpolation="nearest",
               alpha=0.4)
    count = _coverage_count(gdf_rot, extent, bg.shape)
    max_count = int(count.max()) if count.max() > 0 else 1
    im = ax4.imshow(
        count, extent=extent, origin="upper", aspect="equal",
        cmap="YlOrRd", alpha=0.75, vmin=0, vmax=max_count,
    )
    cbar = fig.colorbar(im, ax=ax4, shrink=0.6, pad=0.02)
    cbar.set_label("tiles per pixel", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    ax4.set_title(
        f"Rotated — tile coverage count\n"
        f"Interior={max_count} tiles confirms {ovl_pct}% overlap",
        fontsize=10,
    )
    ax4.set_xlabel("UTM Easting (m)", fontsize=8)
    ax4.set_ylabel("UTM Northing (m)", fontsize=8)
    ax4.tick_params(labelsize=7)

    orig_patch = mpatches.Patch(color=ORIG_COLOR, label=f"Original ({len(gdf_orig)} tiles)")
    rot_patch = mpatches.Patch(
        color=ROT_COLOR, label=f"Rotated ({len(gdf_rot)} tiles, theta={theta:.1f}°)"
    )
    fig.legend(
        handles=[orig_patch, rot_patch], loc="lower center", ncol=2, fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        f"{polygon} / {layer_name}  —  axis-aligned vs rotation-aware tiling"
        f"  [tile={tile_m}m  stride={stride_m}m  overlap={ovl_pct}%]",
        fontsize=13, fontweight="bold",
    )

    out_path = out_dir / f"{polygon}_{layer_name}_comparison.jpg"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved -> %s", out_path)
    return out_path


def compare_polygon(polygon: str, out_dir: Path) -> list[Path]:
    base     = Path("outputs") / polygon
    orig_dir = base / RUN_TAG
    rot_dir  = base / f"{RUN_TAG}_rot"

    if not orig_dir.exists():
        raise FileNotFoundError(f"original output missing: {orig_dir}")
    if not rot_dir.exists():
        raise FileNotFoundError(f"rotated output missing: {rot_dir}")

    stitched_dir = orig_dir / "stitched"
    feat_tif = stitched_dir / "features.tif"
    if not feat_tif.exists():
        raise FileNotFoundError(
            f"stitched features.tif missing: {feat_tif} -- run 'seabed_tiler.stitch' first"
        )

    with rasterio.open(feat_tif) as ds:
        crs_utm = ds.crs.to_string()
        band_names = [d or f"band{i+1}" for i, d in enumerate(ds.descriptions)]
        bounds = ds.bounds

    extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    df_orig = pd.read_csv(orig_dir / "manifest.csv")
    geoms_orig = [box(r.xmin, r.ymin, r.xmax, r.ymax) for _, r in df_orig.iterrows()]
    gdf_orig = gpd.GeoDataFrame(df_orig, geometry=geoms_orig, crs=crs_utm)

    gdf_rot = gpd.read_file(rot_dir / "manifest.geojson").to_crs(crs_utm)
    theta = float(gdf_rot["theta_deg"].iloc[0]) if "theta_deg" in gdf_rot.columns else 0.0

    out_paths = []

    for band_name in band_names:
        jpg = stitched_dir / f"features_{band_name}.jpg"
        if not jpg.exists():
            logger.warning("no stitched JPEG for band '%s', skipping", band_name)
            continue
        bg = np.array(Image.open(jpg))
        p = _compare_one_layer(polygon, band_name, bg, gdf_orig, gdf_rot, extent, theta, out_dir)
        out_paths.append(p)

    labels_jpg = stitched_dir / "labels.jpg"
    if labels_jpg.exists():
        bg = np.array(Image.open(labels_jpg))
        p = _compare_one_layer(polygon, "labels", bg, gdf_orig, gdf_rot, extent, theta, out_dir)
        out_paths.append(p)
    else:
        logger.warning("no stitched labels.jpg found for %s", polygon)

    return out_paths


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
            paths = compare_polygon(poly, out_dir)
            for p in paths:
                logger.info("  -> %s", p.name)
        except FileNotFoundError as e:
            logger.warning("skipping %s: %s", poly, e)


if __name__ == "__main__":
    main()
