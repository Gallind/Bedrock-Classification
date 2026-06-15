"""seabed_forest.pixels: masking of invalid pixels, dedup of tile overlap, subsample cap."""

import numpy as np
from rasterio.transform import from_origin

from seabed_unet.data import TileRecord
from seabed_unet.normalize import compute_stats
from seabed_forest.pixels import build_pixel_table

BANDS = ["backscatter", "bathymetry", "slope"]
MODES = {"backscatter": "per_polygon", "bathymetry": "per_polygon", "slope": "per_polygon"}
NODATA = -9999.0
CLASS_IDS = [1, 2, 3]
IGNORE_LABEL = 0


def _record(features, label, polygon="polygon1", origin=(600000.0, 3600000.0)):
    return TileRecord(
        tile_id="t", polygon=polygon, augmented=False, features=features, label=label,
        transform=from_origin(origin[0], origin[1], 1.0, 1.0), crs="EPSG:32636",
        center_x=origin[0], center_y=origin[1], theta_deg=0.0,
    )


def _stats(records):
    grouped = {}
    for r in records:
        grouped.setdefault(r.polygon, []).append(r.features)
    return compute_stats(grouped, [r.features for r in records], BANDS, MODES, NODATA, (2.0, 98.0))


def test_invalid_pixels_excluded():
    # 4x4 tile: top row label=0 (background), one pixel feature nodata -> all excluded.
    feats = np.random.default_rng(0).uniform(0, 10, size=(3, 4, 4)).astype(np.float32)
    label = np.full((4, 4), 1, dtype=np.uint8)
    label[0, :] = 0                 # 4 background pixels
    feats[0, 1, 1] = NODATA         # 1 nodata pixel
    recs = [_record(feats, label)]
    X, y, coords = build_pixel_table(recs, BANDS, CLASS_IDS, _stats(recs), NODATA, IGNORE_LABEL, MODES)
    assert X.shape == (16 - 4 - 1, 3)        # 11 valid pixels
    assert y.shape == (11,)
    assert set(np.unique(y)).issubset({0, 1, 2})   # channel indices (class_id 1 -> channel 0)
    assert (y == 0).all()                          # every valid pixel is class_id 1 == channel 0


def test_dedup_collapses_overlapping_tiles():
    feats = np.full((3, 4, 4), 5.0, dtype=np.float32)
    label = np.full((4, 4), 1, dtype=np.uint8)
    # Two tiles at the SAME origin => identical world coords for every pixel.
    recs = [_record(feats, label), _record(feats, label)]
    stats = _stats(recs)
    no_dedup, _, _ = build_pixel_table(recs, BANDS, CLASS_IDS, stats, NODATA, IGNORE_LABEL, MODES, dedup=False)
    deduped, _, _ = build_pixel_table(recs, BANDS, CLASS_IDS, stats, NODATA, IGNORE_LABEL, MODES, dedup=True)
    assert no_dedup.shape[0] == 32     # 2 x 16
    assert deduped.shape[0] == 16      # overlap collapsed to one grid


def test_subsample_caps_per_class():
    feats = np.full((3, 10, 10), 5.0, dtype=np.float32)
    label = np.full((10, 10), 1, dtype=np.uint8)
    label[:, 5:] = 3                   # 50 pixels class 1, 50 pixels class 3
    recs = [_record(feats, label)]
    X, y, _ = build_pixel_table(
        recs, BANDS, CLASS_IDS, _stats(recs), NODATA, IGNORE_LABEL, MODES,
        max_pixels_per_class=20, seed=1,
    )
    counts = np.bincount(y, minlength=3)
    assert counts[0] <= 20 and counts[2] <= 20   # channel 0 (class1) and channel 2 (class3)
    assert counts[0] == 20 and counts[2] == 20
