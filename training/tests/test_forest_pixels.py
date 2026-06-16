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


def test_empty_records_returns_correct_shapes():
    X, y, coords = build_pixel_table([], BANDS, CLASS_IDS, {}, NODATA, IGNORE_LABEL, MODES)
    assert X.shape == (0, 3)
    assert y.shape == (0,)
    assert coords.shape == (0, 2)
    assert X.dtype == np.float32 and y.dtype == np.int64


def test_world_coords_handles_rotated_transform():
    from affine import Affine
    # A pure 90-degree rotation + translation: pixel center (col=0,row=0) -> (cx,cy).
    # Affine(a,b,c,d,e,f): x = a*col + b*row + c ; y = d*col + e*row + f.
    # Use a=0,b=1,c=600000 ; d=1,e=0,f=3600000 so center (0.5,0.5) -> (600000.5, 3600000.5).
    transform = Affine(0.0, 1.0, 600000.0, 1.0, 0.0, 3600000.0)
    feats = np.full((3, 2, 2), 5.0, dtype=np.float32)
    label = np.full((2, 2), 1, dtype=np.uint8)
    rec = TileRecord(
        tile_id="t", polygon="polygon1", augmented=False, features=feats, label=label,
        transform=transform, crs="EPSG:32636", center_x=600000.0, center_y=3600000.0, theta_deg=45.0,
    )
    _, _, coords = build_pixel_table([rec], BANDS, CLASS_IDS, _stats([rec]), NODATA, IGNORE_LABEL, MODES)
    # pixel (col=0,row=0): center (0.5,0.5) -> x = 1*0.5 + 600000 = 600000.5 ; y = 1*0.5 + 3600000 = 3600000.5
    assert np.any(np.all(np.isclose(coords, [600000.5, 3600000.5]), axis=1))
