"""watch.py helpers + predict.MapAccumulator (no interactive window)."""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from seabed_unet.data import IGNORE_INDEX, TileRecord
from seabed_unet.predict import MapAccumulator, feature_valid_mask
from seabed_unet.watch import (
    build_truth_map,
    build_backdrop,
    class_overlay,
    masked_label_rgb,
    render_band,
    tile_macro_dice,
    tile_outline_px,
)

NODATA = -9999.0
CLASS_IDS = [1, 2, 3]


def make_record(x0=600000.0, y0=3600000.0, size=8, fill=1.0):
    features = np.full((2, size, size), fill, dtype=np.float32)
    label = np.ones((size, size), dtype=np.uint8)
    return TileRecord(
        tile_id=f"t_{x0:.0f}", polygon="p", augmented=False,
        features=features, label=label,
        transform=from_origin(x0, y0, 1.0, 1.0), crs=CRS.from_epsg(32636),
    )


def test_render_band_shape_and_mask():
    band = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, 0] = False
    rgb = render_band(band, valid, "bathymetry")
    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint8
    assert (rgb[0, 0] == 0).all()          # invalid -> black
    assert rgb[3, 3].sum() > 0


def test_render_band_unknown_band_falls_back_to_gray():
    band = np.full((2, 2), 0.5, dtype=np.float32)
    rgb = render_band(band, np.ones((2, 2), dtype=bool), "hillshade")
    r, g, b = rgb[0, 0]
    assert r == g == b  # grayscale fallback


def test_build_backdrop_covers_tiles_only():
    records = [make_record(x0=600000.0), make_record(x0=600016.0)]  # gap between
    acc = MapAccumulator(records, CLASS_IDS, NODATA)
    stats = {"p": {b: (0.0, 2.0) for b in ("b0", "b1")}}
    modes = {b: "per_polygon" for b in ("b0", "b1")}
    backdrop = build_backdrop(records, acc, ["b0", "b1"], stats, modes, NODATA)
    assert backdrop.shape == (acc.n_rows, acc.n_cols, 3)
    assert backdrop[:, :8].max() > 0       # first tile covered
    assert (backdrop[:, 9:15] == 0).all()  # gap stays black
    # grayscale: channels equal
    assert np.array_equal(backdrop[..., 0], backdrop[..., 1])


def test_class_overlay_paints_only_classified_pixels():
    backdrop = np.full((4, 4, 3), 100, dtype=np.uint8)
    class_map = np.zeros((4, 4), dtype=np.uint8)
    class_map[0, 0] = 1
    out = class_overlay(backdrop, class_map, alpha=1.0)
    assert not np.array_equal(out[0, 0], backdrop[0, 0])   # painted
    assert np.array_equal(out[1:, :], backdrop[1:, :])     # untouched


def test_class_overlay_alpha_blends_with_backdrop():
    backdrop = np.zeros((1, 1, 3), dtype=np.uint8)
    class_map = np.ones((1, 1), dtype=np.uint8)
    full = class_overlay(backdrop, class_map, alpha=1.0)[0, 0].astype(int)
    half = class_overlay(backdrop, class_map, alpha=0.5)[0, 0].astype(int)
    assert (half <= full).all() and half.sum() < full.sum()


def test_tile_outline_px_axis_aligned():
    r = make_record(x0=600004.0, y0=3599996.0)  # 4 px right, 4 px down of map origin
    acc = MapAccumulator([make_record(), r], CLASS_IDS, NODATA)
    corners = tile_outline_px(r, acc)
    assert corners.shape == (4, 2)
    assert corners[0].tolist() == [4.0, 4.0]    # upper-left corner (col, row)
    assert corners[2].tolist() == [12.0, 12.0]  # lower-right corner


def test_masked_label_rgb_blacks_out_invalid():
    ids = np.full((4, 4), 3, dtype=np.uint8)
    valid = np.ones((4, 4), dtype=bool)
    valid[0, :] = False
    rgb = masked_label_rgb(ids, valid)
    assert (rgb[0] == 0).all()
    assert rgb[1:].sum() > 0


def test_tile_macro_dice_perfect_and_ignored():
    target = np.array([[0, 1], [2, IGNORE_INDEX]])
    pred = np.array([[0, 1], [2, 0]])  # value at ignored pixel is irrelevant
    assert tile_macro_dice(pred, target, 3) == pytest.approx(1.0)


def test_tile_macro_dice_fully_unlabeled_is_nan():
    target = np.full((2, 2), IGNORE_INDEX)
    assert np.isnan(tile_macro_dice(np.zeros((2, 2), dtype=int), target, 3))


def test_feature_valid_mask():
    features = np.ones((2, 2, 2), dtype=np.float32)
    features[0, 0, 0] = NODATA
    features[1, 1, 1] = np.nan
    mask = feature_valid_mask(features, NODATA)
    assert mask.tolist() == [[False, True], [True, False]]


def test_accumulator_single_tile_paints_argmax_class():
    r = make_record()
    acc = MapAccumulator([r], CLASS_IDS, NODATA)
    probs = np.zeros((3, 8, 8), dtype=np.float32)
    probs[1] = 1.0  # channel 1 -> class id 2
    acc.add(r, probs)
    out = acc.class_map()
    assert out.shape == (8, 8)
    assert (out == 2).all()


def test_accumulator_invalid_pixels_get_no_vote():
    r = make_record()
    r.features[0, :, :4] = NODATA  # left half invalid
    acc = MapAccumulator([r], CLASS_IDS, NODATA)
    probs = np.full((3, 8, 8), 0.0, dtype=np.float32)
    probs[2] = 1.0
    acc.add(r, probs)
    out = acc.class_map()
    assert (out[:, :4] == 0).all()   # uncovered -> 0
    assert (out[:, 4:] == 3).all()


def test_accumulator_overlap_averages_probs():
    # two tiles offset by half a tile; in the overlap, average of (0.9 class A,
    # 0.6 class B) etc. decides the argmax
    r1 = make_record(x0=600000.0)
    r2 = make_record(x0=600004.0)
    acc = MapAccumulator([r1, r2], CLASS_IDS, NODATA)
    p1 = np.zeros((3, 8, 8), dtype=np.float32)
    p1[0] = 0.9  # strong class id 1
    p2 = np.zeros((3, 8, 8), dtype=np.float32)
    p2[2] = 0.6  # weaker class id 3
    acc.add(r1, p1)
    acc.add(r2, p2)
    out = acc.class_map()
    assert (out[:, :4] == 1).all()      # only tile 1
    assert (out[:, 8:] == 3).all()      # only tile 2
    # overlap: mean(0.45 class1) vs mean(0.3 class3) -> class 1 wins
    assert (out[:, 4:8] == 1).all()


def test_accumulator_incremental_equals_batch():
    """Adding tiles one by one must equal one-shot accumulation (watch == predict)."""
    rng = np.random.default_rng(0)
    records = [make_record(x0=600000.0 + 4 * i) for i in range(3)]
    probs = [rng.random((3, 8, 8)).astype(np.float32) for _ in records]

    a = MapAccumulator(records, CLASS_IDS, NODATA)
    for r, p in zip(records, probs):
        a.add(r, p)

    b = MapAccumulator(records, CLASS_IDS, NODATA)
    for r, p in zip(records, probs):
        b.add(r, p)
        b.class_map()  # interleaved reads must not perturb the accumulation

    assert np.array_equal(a.class_map(), b.class_map())
    assert a.transform == b.transform


def test_build_truth_map_mosaics_labels_with_gaps():
    r1 = make_record(x0=600000.0)
    r2 = make_record(x0=600016.0)  # 8 px gap between tiles
    r2.label[:] = 3
    acc = MapAccumulator([r1, r2], CLASS_IDS, NODATA)
    truth = build_truth_map([r1, r2], acc)
    assert truth.shape == (acc.n_rows, acc.n_cols)
    assert (truth[:, :8] == 1).all()       # r1 labels
    assert (truth[:, 8:16] == 0).all()     # gap stays unlabeled
    assert (truth[:, 16:] == 3).all()      # r2 labels
    assert set(np.unique(truth).tolist()) <= {0, 1, 3}  # nearest: no interpolated ids
