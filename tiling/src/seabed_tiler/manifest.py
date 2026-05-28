"""Write the tile manifest (CSV + GeoJSON) and a grid-overlap preview.

CSV feeds ML data loaders; the GeoJSON files (reprojected to WGS84 so they open anywhere)
let you drop the grid into QGIS and eyeball that neighbours overlap by 50 %.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


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
