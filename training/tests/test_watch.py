"""watch.py helpers + predict.MapAccumulator (no interactive window)."""

import numpy as np
import pytest
from rasterio.crs import CRS
from rasterio.transform import from_origin

from seabed_unet.data import IGNORE_INDEX, TileRecord
from seabed_unet.predict import MapAccumulator, feature_valid_mask
from seabed_unet.watch import compose_input_rgb, masked_label_rgb, tile_macro_dice

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


def test_compose_input_rgb_three_bands():
    inputs = np.stack([np.full((4, 4), v, dtype=np.float32) for v in (0.0, 0.5, 1.0)])
    rgb = compose_input_rgb(inputs)
    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint8
    assert rgb[0, 0].tolist() == [0, 127, 255]


def test_compose_input_rgb_two_bands_pads_blue():
    inputs = np.ones((2, 4, 4), dtype=np.float32)
    rgb = compose_input_rgb(inputs)
    assert rgb[..., 2].max() == 0  # missing band -> zero channel


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
