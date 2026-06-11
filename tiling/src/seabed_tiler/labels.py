"""Turn class shapefile(s) into a single-band uint8 label raster on the master grid.

Two label encodings are supported (see ``LabelsConfig.kind``):

* ``shapefile`` (polygon1) — one file whose noisy NAME field (mixed case, stray spaces,
  e.g. ``"Class 2-  shallow rock"`` vs ``"class2 -shallow rock"``) is normalized and
  matched against ordered regex rules; "shallow" is matched before "rock" so shallow
  rock is not swallowed by the rock rule.
* ``shapefile_per_class`` (polygons 3/4/5) — one or more shapefiles per class, burned in
  priority order so a higher-priority class wins on overlap. LineString labels
  (polygon3) are closed into polygons first; features that cannot form an area are
  dropped and reported.
"""

from __future__ import annotations

import logging
import re

import geopandas as gpd
import numpy as np
from rasterio.features import rasterize
from shapely import make_valid
from shapely.geometry import Polygon
from shapely.ops import polygonize as shp_polygonize

from .config import Config

logger = logging.getLogger(__name__)


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


def _iter_polys(geom) -> list:
    """Flatten a (possibly multi/collection) geometry to its non-empty polygons."""
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom]
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        out: list = []
        for part in geom.geoms:
            out.extend(_iter_polys(part))
        return out
    return []


def _feature_to_polygons(geom, polygonize: bool) -> list:
    """Convert one shapefile feature to polygons, repairing invalid geometry.

    Polygon/MultiPolygon features are validated and returned. LineString features are
    closed into polygons only when ``polygonize`` is set and the ring closes (else []).
    """
    if geom is None or geom.is_empty:
        return []
    if geom.geom_type in ("Polygon", "MultiPolygon", "GeometryCollection"):
        return [p for p in _iter_polys(make_valid(geom)) if p.area > 0]
    if polygonize and geom.geom_type in ("LineString", "LinearRing"):
        coords = list(geom.coords)
        if len(coords) >= 4 and coords[0] == coords[-1]:
            polys = [p for p in _iter_polys(make_valid(Polygon(coords))) if p.area > 0]
            if polys:
                return polys
        # Fall back to shapely's polygonizer (handles some self-touching rings).
        return [p for p in shp_polygonize([geom]) if p.area > 0]
    return []


def build_label_per_class(cfg: Config, transform, crs, shape) -> np.ndarray:
    """Rasterize per-class shapefiles, burning classes in priority order (low -> high)."""
    nodata = cfg.output.label_nodata
    classes = cfg.labels.classes
    class_files = cfg.labels.class_files or {}
    priority = cfg.labels.priority or list(classes)
    polygonize = cfg.labels.polygonize

    shapes: list[tuple] = []
    for class_name in priority:
        class_id = classes[class_name]
        kept = dropped = 0
        for fname in class_files.get(class_name, []):
            gdf = gpd.read_file(cfg.src_path / fname)
            gdf = gdf.set_crs(cfg.crs) if gdf.crs is None else gdf.to_crs(cfg.crs)
            for idx, geom in enumerate(gdf.geometry):
                polys = _feature_to_polygons(geom, polygonize)
                if polys:
                    shapes.extend((p, class_id) for p in polys)
                    kept += 1
                else:
                    dropped += 1
                    gtype = "empty" if geom is None or geom.is_empty else geom.geom_type
                    logger.info(f"    [drop] {class_name}: {fname}[{idx}] ({gtype}) not a closed area")
        if dropped:
            logger.info(f"    {class_name}: kept {kept}, dropped {dropped} feature(s)")

    if not shapes:
        return np.full(shape, nodata, dtype=np.uint8)

    # rasterize burns shapes in order; later (higher-priority) shapes win on overlap.
    return rasterize(
        shapes,
        out_shape=shape,
        transform=transform,
        fill=nodata,
        dtype="uint8",
        all_touched=False,
    )


def build_label_array(cfg: Config, transform, crs, shape) -> np.ndarray:
    """Rasterize labeled polygons onto the master grid; unlabeled cells get label_nodata."""
    if cfg.labels.kind == "shapefile_per_class":
        return build_label_per_class(cfg, transform, crs, shape)

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
