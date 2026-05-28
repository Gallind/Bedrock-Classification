"""Shared helpers for turning float feature / uint8 label rasters into viewable JPEGs.

Keeps geolocation by writing an ESRI world file (.jgw) + .prj next to each JPEG, matching
the format of the original survey data so the JPEGs still line up in QGIS.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# Class id -> RGB. Diverging palette so the three seabed classes are easy to tell apart.
LABEL_COLORS = {
    0: (0, 0, 0),         # background / unlabeled
    1: (202, 0, 32),      # rock          - red
    2: (244, 165, 130),   # shallow_rock  - salmon
    3: (5, 113, 176),     # sand          - blue
}
_UNKNOWN_COLOR = (255, 255, 255)


def normalize_band(arr, nodata, vmin=None, vmax=None, p_low=2, p_high=98):
    """Scale a float band to uint8 [0,255]; nodata/NaN -> 0 (black).

    If vmin/vmax are not given they are taken from the p_low/p_high percentiles of the
    valid pixels (2-98 by default), matching scripts/render_python.py. Returns
    (uint8_array, valid_mask).
    """
    a = arr.astype("float32")
    mask = (a != nodata) & ~np.isnan(a)
    if vmin is None or vmax is None:
        if mask.any():
            vmin, vmax = np.percentile(a[mask], [p_low, p_high])
        else:
            vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    scaled = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
    u8 = (scaled * 255).astype("uint8")
    u8[~mask] = 0
    return u8, mask


def label_to_rgb(arr) -> np.ndarray:
    """Map a uint8 class-id raster to an RGB image using LABEL_COLORS."""
    rgb = np.empty((*arr.shape, 3), dtype="uint8")
    rgb[...] = _UNKNOWN_COLOR
    for class_id, color in LABEL_COLORS.items():
        rgb[arr == class_id] = color
    return rgb


def write_worldfile(jpg_path, transform) -> None:
    """Write the ESRI world file (.jgw): pixel size + center of the upper-left pixel."""
    a, b, d, e = transform.a, transform.b, transform.d, transform.e
    c = transform.c + a / 2.0       # x of the upper-left pixel center
    f = transform.f + e / 2.0       # y of the upper-left pixel center
    Path(jpg_path).with_suffix(".jgw").write_text(f"{a}\n{d}\n{b}\n{e}\n{c}\n{f}\n")


def write_prj(jpg_path, crs) -> None:
    """Write the CRS as WKT next to the JPEG."""
    Path(jpg_path).with_suffix(".prj").write_text(crs.to_wkt())
