"""Write the tile manifest (CSV + GeoJSON) and a grid-overlap preview.

CSV feeds ML data loaders; the GeoJSON files (reprojected to WGS84 so they open anywhere)
let you drop the grid into QGIS and eyeball that neighbours overlap by 50 %.
"""

from __future__ import annotations

import math as _math
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box, Polygon as _Polygon
from pyproj import Transformer as _Transformer


def write_manifest(rows: list[dict], out_dir: Path, crs) -> pd.DataFrame:
    """Write manifest.csv (always) and manifest.geojson (written tiles only)."""
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "manifest.csv", index=False)

    if not df.empty:
        geoms = [box(r["xmin"], r["ymin"], r["xmax"], r["ymax"]) for r in rows]
        gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs).to_crs("EPSG:4326")
        gdf.to_file(out_dir / "manifest.geojson", driver="GeoJSON")

    return df


def write_grid_preview(windows: list, out_dir: Path, crs) -> None:
    """Write every candidate window (pre-filter) so the overlap pattern is visible."""
    if not windows:
        return
    geoms = [box(w.xmin, w.ymin, w.xmax, w.ymax) for w in windows]
    data = [{"row": w.row, "col": w.col} for w in windows]
    gdf = gpd.GeoDataFrame(data, geometry=geoms, crs=crs).to_crs("EPSG:4326")
    gdf.to_file(out_dir / "grid_preview.geojson", driver="GeoJSON")


def _tile_corners_utm(
    u_origin: float, v_origin: float, theta_deg: float, res: float, tile_px: int
) -> list[tuple[float, float]]:
    """Compute the 4 UTM corner coords of a rotated tile as (x, y) pairs.

    Corners in order: top-left, top-right, bottom-right, bottom-left.
    """
    theta = _math.radians(theta_deg)
    c, s = _math.cos(theta), _math.sin(theta)
    ox = u_origin * c - v_origin * s
    oy = u_origin * s + v_origin * c
    size = res * tile_px
    tl = (ox, oy)
    tr = (ox + size * c, oy + size * s)
    br = (ox + size * c + size * s, oy + size * s - size * c)
    bl = (ox + size * s, oy - size * c)
    return [tl, tr, br, bl]


def write_rotated_manifest(
    rows: list[dict],
    out_dir: Path,
    crs,
    res: float,
    tile_px: int,
) -> None:
    """Write manifest.csv and manifest.geojson for rotation-aware tiles.

    The GeoJSON geometry for each tile is the actual 4-corner polygon in WGS84,
    not an axis-aligned bounding box.
    """
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "manifest.csv", index=False)

    if df.empty:
        return

    crs_str = str(crs) if not isinstance(crs, str) else crs
    transformer = _Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)

    geoms = []
    for r in rows:
        corners_utm = _tile_corners_utm(r["u_origin"], r["v_origin"], r["theta_deg"], res, tile_px)
        corners_wgs84 = [transformer.transform(x, y) for x, y in corners_utm]
        ring = corners_wgs84 + [corners_wgs84[0]]
        geoms.append(_Polygon(ring))

    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs="EPSG:4326")
    gdf.to_file(out_dir / "manifest.geojson", driver="GeoJSON")
