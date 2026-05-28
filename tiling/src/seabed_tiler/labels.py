"""Turn the class shapefile into a single-band uint8 label raster on the master grid.

The shapefile's NAME field is noisy (mixed case, stray spaces, e.g. ``"Class 2-  shallow
rock"`` vs ``"class2 -shallow rock"``), so names are normalized and matched against
ordered regex rules. "shallow" is matched before "rock" so shallow rock is not swallowed
by the rock rule.
"""

from __future__ import annotations

import re

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize

from .config import Config


def normalize_name(name) -> str:
    """Lowercase, strip, and collapse internal whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", str(name).strip().lower())


def classify(name, rules, classes) -> int | None:
    """Return the class id for ``name`` using the first matching rule, else ``None``."""
    normalized = normalize_name(name)
    for rule in rules:
        if re.search(rule.pattern, normalized):
            return classes.get(rule.class_)
    return None


def build_label_array(cfg: Config, transform, crs, shape) -> np.ndarray:
    """Rasterize labeled polygons onto the master grid; unlabeled cells get label_nodata."""
    gdf = gpd.read_file(cfg.labels_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(cfg.crs)
    else:
        gdf = gdf.to_crs(cfg.crs)

    field = cfg.labels.name_field
    nodata = cfg.output.label_nodata

    shapes = []
    for geom, name in zip(gdf.geometry, gdf[field]):
        if geom is None or geom.is_empty:
            continue
        class_id = classify(name, cfg.labels.rules, cfg.labels.classes)
        if class_id is not None:
            shapes.append((geom, class_id))

    if not shapes:
        return np.full(shape, nodata, dtype=np.uint8)

    return rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=nodata,
        dtype="uint8",
        all_touched=False,
    )
