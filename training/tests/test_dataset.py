"""dataset.py: tensor shapes/dtypes, D4 pair-consistency, determinism."""

import numpy as np
import torch

from conftest import BANDS, NODATA

from seabed_unet.data import IGNORE_INDEX, TileRecord
from seabed_unet.dataset import TileDataset

CLASS_IDS = [1, 2, 3]


def make_record(size=8) -> TileRecord:
    rng = np.random.default_rng(0)
    features = rng.uniform(0, 100, (3, size, size)).astype(np.float32)
    label = rng.integers(0, 4, (size, size)).astype(np.uint8)
    return TileRecord("t_r000_c000", "polygon1", False, features, label)


def make_stats():
    return {"polygon1": {b: (0.0, 100.0) for b in BANDS}}


def test_tensor_shapes_and_dtypes():
    ds = TileDataset([make_record()], BANDS, CLASS_IDS, make_stats(), NODATA, 0)
    x, y = ds[0]
    assert x.shape == (3, 8, 8) and x.dtype == torch.float32
    assert y.shape == (8, 8) and y.dtype == torch.int64
    assert x.min() >= 0.0 and x.max() <= 1.0
    assert set(y.unique().tolist()) <= {IGNORE_INDEX, 0, 1, 2}


def test_no_augment_is_deterministic_identity():
    ds = TileDataset([make_record()], BANDS, CLASS_IDS, make_stats(), NODATA, 0, augment=False)
    x1, y1 = ds[0]
    x2, y2 = ds[0]
    assert torch.equal(x1, x2) and torch.equal(y1, y2)


def test_d4_augment_moves_features_and_target_together():
    record = make_record()
    ds = TileDataset([record], BANDS, CLASS_IDS, make_stats(), NODATA, 0, augment=True, seed=1)
    base = TileDataset([record], BANDS, CLASS_IDS, make_stats(), NODATA, 0, augment=False)
    x0, y0 = base[0]
    # Across several draws, augmented pairs must stay co-registered: the same
    # D4 op maps base (x, y) onto the augmented (x, y).
    from seabed_tiler.augment import D4_OPS, augment_pair

    for _ in range(8):
        xa, ya = ds[0]
        matched = False
        for op in D4_OPS:
            xo, yo = augment_pair(x0.numpy(), y0.numpy(), op)
            if np.array_equal(xo, xa.numpy()) and np.array_equal(yo, ya.numpy()):
                matched = True
                break
        assert matched, "augmented pair is not any D4 transform of the base pair"


def test_augment_seed_reproducible():
    record = make_record()
    a = TileDataset([record], BANDS, CLASS_IDS, make_stats(), NODATA, 0, augment=True, seed=7)
    b = TileDataset([record], BANDS, CLASS_IDS, make_stats(), NODATA, 0, augment=True, seed=7)
    for _ in range(4):
        xa, ya = a[0]
        xb, yb = b[0]
        assert torch.equal(xa, xb) and torch.equal(ya, yb)
