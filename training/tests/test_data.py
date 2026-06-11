"""data.py: band selection, split contract enforcement, target encoding."""

import numpy as np
import pytest

from conftest import BANDS, NODATA, SIZE, make_run_dir, make_tile

from seabed_unet.config import Config
from seabed_unet.data import (
    IGNORE_INDEX,
    encode_target,
    features_by_polygon,
    load_run_records,
    load_split_records,
)


def make_cfg(base_dir, use_augmented=True, bands=None):
    cfg = Config(
        name="t",
        bands=bands or list(BANDS),
        split={
            "train": ["polygon1"],
            "val": ["polygon3"],
            "test": ["polygon4"],
            "use_augmented_for_train": use_augmented,
        },
    )
    cfg.base_dir = base_dir
    return cfg


def test_reads_requested_bands_by_description(synth_dataset):
    cfg = make_cfg(synth_dataset, bands=["bathymetry", "slope"])
    records = load_run_records(
        cfg.rot_dir("polygon1"), "polygon1", cfg.bands, cfg.base_dir, augmented=False
    )
    assert len(records) == 1
    feats = records[0].features
    assert feats.shape == (2, SIZE, SIZE)
    # polygon1 fills are (10, 11, 12) for (backscatter, bathymetry, slope)
    assert feats[0, 0, 0] == 11.0  # bathymetry
    assert feats[1, 0, 0] == 12.0  # slope


def test_missing_band_raises(synth_dataset):
    cfg = make_cfg(synth_dataset, bands=["hillshade"])
    with pytest.raises(ValueError, match="hillshade"):
        load_run_records(
            cfg.rot_dir("polygon1"), "polygon1", cfg.bands, cfg.base_dir, augmented=False
        )


def test_split_assignment_follows_polygon_lists(synth_dataset):
    splits = load_split_records(make_cfg(synth_dataset))
    assert {r.polygon for r in splits["train"]} == {"polygon1"}
    assert {r.polygon for r in splits["val"]} == {"polygon3"}
    assert {r.polygon for r in splits["test"]} == {"polygon4"}


def test_augmented_tiles_only_in_train(synth_dataset):
    splits = load_split_records(make_cfg(synth_dataset))
    # _rotaug dirs exist for ALL three polygons; only train may use them
    assert any(r.augmented for r in splits["train"])
    assert not any(r.augmented for r in splits["val"])
    assert not any(r.augmented for r in splits["test"])


def test_use_augmented_false_excludes_rotaug(synth_dataset):
    splits = load_split_records(make_cfg(synth_dataset, use_augmented=False))
    assert len(splits["train"]) == 1
    assert not any(r.augmented for r in splits["train"])


def test_missing_manifest_raises_with_pointer_to_guide(tmp_path):
    cfg = make_cfg(tmp_path)
    with pytest.raises(FileNotFoundError, match="TRAINING_DATA_SETUP"):
        load_split_records(cfg)


def test_encode_target_maps_class_ids_to_channels():
    label = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    features = np.ones((3, 2, 2), dtype=np.float32)
    target = encode_target(label, features, [1, 2, 3], NODATA, 0)
    assert target.tolist() == [[IGNORE_INDEX, 0], [1, 2]]


def test_encode_target_ignores_feature_invalid_pixels():
    label = np.ones((2, 2), dtype=np.uint8)  # all rock
    features = np.ones((2, 2, 2), dtype=np.float32)
    features[0, 0, 0] = NODATA      # band 0 nodata at (0,0)
    features[1, 1, 1] = np.nan      # band 1 NaN at (1,1)
    target = encode_target(label, features, [1, 2, 3], NODATA, 0)
    assert target[0, 0] == IGNORE_INDEX
    assert target[1, 1] == IGNORE_INDEX
    assert target[0, 1] == 0 and target[1, 0] == 0


def test_features_by_polygon_excludes_augmented(synth_dataset):
    splits = load_split_records(make_cfg(synth_dataset))
    grouped = features_by_polygon(splits)
    assert set(grouped) == {"polygon1", "polygon3", "polygon4"}
    assert all(len(arrays) == 1 for arrays in grouped.values())  # base tiles only


def test_spatial_blocks_mode_end_to_end(tmp_path):
    # one polygon, 40 base tiles strung along x at the real 64 m stride
    tiles, centers = {}, {}
    for i in range(40):
        tiles[f"t_{i:03d}"] = make_tile([10.0, 11.0, 12.0], label_value=1)
        centers[f"t_{i:03d}"] = (600000 + i * 64.0, 3600000.0)
    make_run_dir(tmp_path, "polygon1", "_rot", tiles, theta_deg=0.0, centers=centers)
    aug_tiles = {f"a_{i:03d}": make_tile([10.0, 11.0, 12.0], label_value=2) for i in range(0, 40, 2)}
    aug_centers = {f"a_{i:03d}": (600032 + i * 64.0, 3600000.0) for i in range(0, 40, 2)}
    make_run_dir(tmp_path, "polygon1", "_rotaug", aug_tiles, theta_deg=0.0, centers=aug_centers)

    cfg = Config(
        name="t",
        bands=list(BANDS),
        split={"mode": "spatial_blocks", "polygons": ["polygon1"],
               "fractions": (0.7, 0.15, 0.15), "buffer_m": 128.0},
    )
    cfg.base_dir = tmp_path
    splits = load_split_records(cfg)
    assert splits["train"] and splits["val"] and splits["test"]
    assert not any(r.augmented for r in splits["val"] + splits["test"])
    assert any(r.augmented for r in splits["train"])
    assert all(r.center_x > 0 for v in splits.values() for r in v)
