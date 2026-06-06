# tiling/src/seabed_tiler/rotation.py
"""Rotation-aware tile geometry utilities.

Computes the minimum-area bounding rectangle (MBR) of a survey polygon's annotation
footprint, builds a rotated tile grid aligned to the MBR's long axis, and constructs
per-tile affine transforms for warping data from a North-up master grid.

Affine convention (rasterio/affine):  x = a*col + b*row + c,  y = d*col + e*row + f
For a raster whose column axis makes angle theta with UTM East (CCW positive):
    a = res * cos(theta)   b = res * sin(theta)   c = origin_x_utm
    d = res * sin(theta)   e = -res * cos(theta)  f = origin_y_utm
At theta=0 this collapses to the standard North-up affine (a=res, b=0, d=0, e=-res).

(u_origin, v_origin) stored in RotatedTileWindow are UTM coordinates expressed in the
rotated frame (u along the long axis, v perpendicular). They are large UTM-scale values,
not MBR-relative offsets.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from pathlib import Path

from affine import Affine
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

logger = logging.getLogger(__name__)

# Rotations within this many degrees of axis-aligned (0 or 90) are treated as
# axis-aligned to avoid a costly warp pass for negligible benefit.
_MIN_THETA_DEG = 5.0


@dataclass(frozen=True)
class RotatedTileWindow:
    """One tile's position in the rotated grid.

    u_origin, v_origin are UTM coords in the rotated frame: u along the MBR long
    axis (CCW angle theta from UTM East), v perpendicular. Both are large UTM-scale
    values, not offsets from the MBR centroid.
    """
    row: int
    col: int
    u_origin: float
    v_origin: float
    theta: float  # radians, CCW from UTM East


def build_tile_affine(window: RotatedTileWindow, res: float) -> Affine:
    """Construct the rasterio affine transform for a rotated tile.

    Maps pixel (col, row) to UTM (x, y). Moving one pixel right steps along
    the MBR long axis (u direction); moving one pixel down steps along -v.
    """
    c = math.cos(window.theta)
    s = math.sin(window.theta)
    # Convert (u_origin, v_origin) from rotated frame to UTM:
    # x = u*cos - v*sin,  y = u*sin + v*cos
    ox = window.u_origin * c - window.v_origin * s
    oy = window.u_origin * s + window.v_origin * c
    return Affine(res * c, res * s, ox,
                  res * s, -res * c, oy)


def compute_label_footprint(shapefile_paths: list[Path]) -> Polygon:
    """Return the convex hull of the union of all annotation polygons.

    shapefile_paths: list of Path objects pointing to label shapefiles.
    All geometries are reprojected to EPSG:32636 before unioning.

    Raises ValueError if no valid geometries are found across all files.
    """
    all_geoms = []
    crs_target = "EPSG:32636"
    for shp in shapefile_paths:
        gdf = gpd.read_file(shp)
        if gdf.crs is None:
            gdf = gdf.set_crs(crs_target)
        else:
            gdf = gdf.to_crs(crs_target)
        valid = gdf[gdf.geometry.is_valid & ~gdf.geometry.is_empty]
        all_geoms.extend(valid.geometry.tolist())

    if not all_geoms:
        raise ValueError("no valid annotation geometries found in provided shapefiles")

    union = unary_union(all_geoms)
    footprint = union.convex_hull
    if not isinstance(footprint, Polygon):
        raise ValueError(
            f"label footprint resolved to {type(footprint).__name__}, not a Polygon; "
            "annotation shapefiles may contain only degenerate (collinear) geometries"
        )
    logger.info("label footprint: %d source geometries, area=%.1f m^2", len(all_geoms), footprint.area)
    return footprint


def minimum_bounding_rect(
    polygon: Polygon,
) -> tuple[Polygon, float, np.ndarray]:
    """Compute the minimum-area bounding rectangle of a shapely Polygon.

    Returns:
        mbr: the MBR as a Polygon (4 vertices + closing vertex).
        theta: rotation angle in radians of the MBR's LONG axis from UTM East (CCW).
               Clamped to 0.0 when |theta_deg| < _MIN_THETA_DEG or > 90-_MIN_THETA_DEG
               (axis-aligned fallback avoids a costly warp for negligible benefit).
        corners: (4, 2) float64 array of MBR corner UTM coords.
    """
    mbr = polygon.minimum_rotated_rectangle
    coords = np.array(mbr.exterior.coords)[:-1]  # drop closing repeat -> (4, 2)

    # Compute all 4 edge vectors; the MBR has 2 distinct orientations (edges 0 and 1).
    edge_vecs = np.diff(np.vstack([coords, coords[0]]), axis=0)  # (4, 2)
    edge_lengths = np.linalg.norm(edge_vecs, axis=1)

    # Pick the long edge (index 0 or 1 — adjacent edges are perpendicular).
    long_idx = int(np.argmax(edge_lengths[:2]))
    dx, dy = edge_vecs[long_idx]
    theta = math.atan2(dy, dx)

    # Normalise to [-pi/2, pi/2] — choose the shorter rotation direction.
    if theta > math.pi / 2:
        theta -= math.pi
    elif theta < -math.pi / 2:
        theta += math.pi

    deg = abs(math.degrees(theta))
    if deg < _MIN_THETA_DEG or deg > (90.0 - _MIN_THETA_DEG):
        logger.warning(
            "MBR theta=%.1f deg is near axis-aligned (threshold=%.0f deg); using theta=0",
            math.degrees(theta), _MIN_THETA_DEG,
        )
        theta = 0.0

    logger.info("MBR long-axis theta=%.2f deg, MBR area=%.1f m^2", math.degrees(theta), mbr.area)
    return mbr, theta, coords
