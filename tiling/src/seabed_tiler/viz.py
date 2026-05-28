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

# Default JPEG rendering per band, chosen to mimic the original DataBase renders.
# Bands not listed here render as grayscale. A polygon config can override these.
BAND_STYLE = {
    "bathymetry": {"cmap": "summer", "hillshade": True, "vert_exag": 5.0},
    "slope": {"cmap": "YlOrRd", "hillshade": False, "vert_exag": 1.0},
}


def _get_cmap(name):
    import matplotlib

    try:
        return matplotlib.colormaps[name]
    except (KeyError, AttributeError):
        import matplotlib.cm as cm

        return cm.get_cmap(name)


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


def colorize(arr, nodata, cmap_name, vmin=None, vmax=None, hillshade=False,
             dx=0.5, vert_exag=5.0, p_low=2, p_high=98) -> np.ndarray:
    """Map a float band to an RGB image via a matplotlib colormap (nodata/NaN -> black).

    With ``hillshade=True`` the colormap is blended with relief shading computed from the
    surface itself (matplotlib LightSource), reproducing hillshaded elevation renders.
    ``dx`` is the cell size in meters; ``vert_exag`` exaggerates the relief.
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

    cmap = _get_cmap(cmap_name)
    if hillshade:
        from matplotlib.colors import LightSource, Normalize

        fill = float(a[mask].mean()) if mask.any() else 0.0
        surface = np.where(mask, a, fill)
        light = LightSource(azdeg=315, altdeg=45)
        shaded = light.shade(
            surface, cmap=cmap, norm=Normalize(vmin, vmax),
            blend_mode="soft", vert_exag=vert_exag, dx=dx, dy=dx,
        )
        rgb = (shaded[..., :3] * 255).astype("uint8")
    else:
        normed = np.clip((a - vmin) / (vmax - vmin), 0.0, 1.0)
        rgb = (cmap(normed)[..., :3] * 255).astype("uint8")

    rgb[~mask] = 0
    return rgb


def resolve_styles(config_path=None) -> dict:
    """Band -> render style dict, starting from BAND_STYLE and overlaying a config.

    A layer with an explicit ``cmap`` overrides the default; a layer with ``to_gray``
    and no ``cmap`` forces grayscale (style removed).
    """
    styles = {k: dict(v) for k, v in BAND_STYLE.items()}
    if config_path:
        from pathlib import Path

        if Path(config_path).exists():
            from .config import load_config

            cfg = load_config(config_path)
            for layer in cfg.layers:
                if layer.cmap:
                    styles[layer.name] = {
                        "cmap": layer.cmap,
                        "hillshade": layer.hillshade,
                        "vert_exag": layer.vert_exag,
                    }
                elif layer.to_gray:
                    styles.pop(layer.name, None)
    return styles


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
