# tiling/tests/test_augment.py
"""Tests for the D4 (dihedral group) pair-augmentation module.

The augmentation API operates on the (features, label) pair as one atomic unit:
features (B, H, W) float32 and label (H, W) uint8 must always transform together,
mirroring the bundle coupling of the raw survey data.
"""
import numpy as np
import pytest

from seabed_tiler.augment import D4_OPS, augment_pair, inverse_op, random_d4


def _make_pair(size: int = 8, bands: int = 3) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    features = rng.normal(size=(bands, size, size)).astype("float32")
    label = rng.integers(0, 4, size=(size, size)).astype("uint8")
    return features, label


def test_d4_has_eight_ops():
    assert len(D4_OPS) == 8
    assert len(set(D4_OPS)) == 8
    assert "identity" in D4_OPS


def test_identity_returns_equal_arrays():
    features, label = _make_pair()
    f2, l2 = augment_pair(features, label, "identity")
    np.testing.assert_array_equal(f2, features)
    np.testing.assert_array_equal(l2, label)


def test_rot90_semantics_known_array():
    """rot90 must be a counter-clockwise quarter turn (numpy rot90 convention)
    applied identically to every band and the label."""
    features = np.array([[[1, 2],
                          [3, 4]]], dtype="float32")  # (1, 2, 2)
    label = np.array([[1, 2],
                      [3, 4]], dtype="uint8")
    f2, l2 = augment_pair(features, label, "rot90")
    expected = np.array([[2, 4],
                         [1, 3]])
    np.testing.assert_array_equal(f2[0], expected.astype("float32"))
    np.testing.assert_array_equal(l2, expected.astype("uint8"))


@pytest.mark.parametrize("op", ["identity", "rot90", "rot180", "rot270",
                                "fliph", "flipv", "transpose", "anti_transpose"])
def test_op_then_inverse_restores_pair_bit_exact(op):
    features, label = _make_pair()
    f2, l2 = augment_pair(features, label, op)
    f3, l3 = augment_pair(f2, l2, inverse_op(op))
    np.testing.assert_array_equal(f3, features)
    np.testing.assert_array_equal(l3, label)


@pytest.mark.parametrize("op", ["rot90", "rot180", "rot270",
                                "fliph", "flipv", "transpose", "anti_transpose"])
def test_features_and_label_move_together(op):
    """A marker pixel at an asymmetric position must land at the same (row, col)
    in every feature band and in the label -- the silent-corruption check."""
    size = 8
    features = np.zeros((2, size, size), dtype="float32")
    label = np.zeros((size, size), dtype="uint8")
    features[:, 1, 5] = 7.0
    label[1, 5] = 3
    f2, l2 = augment_pair(features, label, op)
    label_pos = np.argwhere(l2 == 3)
    assert label_pos.shape == (1, 2)
    for band in range(2):
        band_pos = np.argwhere(f2[band] == 7.0)
        np.testing.assert_array_equal(band_pos, label_pos)


def test_dtypes_and_shapes_preserved():
    features, label = _make_pair(size=8, bands=3)
    f2, l2 = augment_pair(features, label, "rot270")
    assert f2.dtype == np.float32
    assert l2.dtype == np.uint8
    assert f2.shape == features.shape
    assert l2.shape == label.shape
    assert f2.flags["C_CONTIGUOUS"]
    assert l2.flags["C_CONTIGUOUS"]


def test_rejects_unknown_op():
    features, label = _make_pair()
    with pytest.raises(ValueError, match="unknown D4 op"):
        augment_pair(features, label, "brightness")


def test_rejects_mismatched_shapes():
    features, label = _make_pair(size=8)
    bad_label = np.zeros((4, 4), dtype="uint8")
    with pytest.raises(ValueError, match="shape"):
        augment_pair(features, bad_label, "rot90")


def test_rejects_non_square_tiles():
    features = np.zeros((1, 4, 8), dtype="float32")
    label = np.zeros((4, 8), dtype="uint8")
    with pytest.raises(ValueError, match="square"):
        augment_pair(features, label, "rot90")


def test_rejects_wrong_dimensionality():
    label = np.zeros((8, 8), dtype="uint8")
    with pytest.raises(ValueError, match="features"):
        augment_pair(np.zeros((8, 8), dtype="float32"), label, "identity")
    with pytest.raises(ValueError, match="label"):
        augment_pair(np.zeros((1, 8, 8), dtype="float32"),
                     np.zeros((1, 8, 8), dtype="uint8"), "identity")


def test_inverse_op_covers_all_ops():
    for op in D4_OPS:
        assert inverse_op(op) in D4_OPS


def test_random_d4_deterministic_with_seed():
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    draws1 = [random_d4(rng1) for _ in range(20)]
    draws2 = [random_d4(rng2) for _ in range(20)]
    assert draws1 == draws2
    assert all(op in D4_OPS for op in draws1)
    assert len(set(draws1)) > 1  # not stuck on one op
