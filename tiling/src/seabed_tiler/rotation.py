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

from affine import Affine

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
