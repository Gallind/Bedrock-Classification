"""splits.py: block assignment, buffer guarantees, augmentation containment."""

import math

import numpy as np
import pytest

from seabed_unet.data import TileRecord
from seabed_unet.splits import assign_spatial_blocks

TILE_M = 128.0
FRACTIONS = (0.7, 0.15, 0.15)


def make_record(polygon, cx, cy, theta_deg=0.0, augmented=False, tid=None):
    return TileRecord(
        tile_id=tid or f"{polygon}_{cx:.0f}_{cy:.0f}{'_aug' if augmented else ''}",
        polygon=polygon,
        augmented=augmented,
        features=np.zeros((1, 2, 2), dtype=np.float32),
        label=np.zeros((2, 2), dtype=np.uint8),
        center_x=cx,
        center_y=cy,
        theta_deg=theta_deg,
    )


def strip_records(polygon, n=100, stride=64.0, theta_deg=0.0, augmented=False, y0=0.0):
    """n tiles along the survey long axis at the real 64 m stride."""
    theta = math.radians(theta_deg)
    return [
        make_record(
            polygon,
            i * stride * math.cos(theta) - y0 * math.sin(theta),
            i * stride * math.sin(theta) + y0 * math.cos(theta),
            theta_deg,
            augmented,
            tid=f"{polygon}_{i}{'_a' if augmented else ''}",
        )
        for i in range(n)
    ]


def u_of(r):
    theta = math.radians(r.theta_deg)
    return r.center_x * math.cos(theta) + r.center_y * math.sin(theta)


def test_fractions_roughly_respected():
    splits = assign_spatial_blocks(strip_records("p1", n=200), FRACTIONS, buffer_m=TILE_M)
    total = sum(len(v) for v in splits.values())
    assert len(splits["train"]) / total == pytest.approx(0.7, abs=0.08)
    assert len(splits["val"]) / total == pytest.approx(0.15, abs=0.08)
    assert len(splits["test"]) / total == pytest.approx(0.15, abs=0.08)


@pytest.mark.parametrize("theta_deg", [0.0, -31.0])
def test_no_pixel_overlap_across_splits(theta_deg):
    """Kept tiles in different splits must be > tile diagonal apart along u."""
    records = strip_records("p1", n=200, theta_deg=theta_deg)
    splits = assign_spatial_blocks(records, FRACTIONS, buffer_m=TILE_M)
    diagonal = TILE_M * math.sqrt(2)
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        min_gap = min(
            abs(u_of(ra) - u_of(rb)) for ra in splits[a] for rb in splits[b]
        )
        assert min_gap > diagonal, f"{a}/{b} separated by only {min_gap:.0f} m"


def test_buffer_strip_tiles_are_dropped():
    records = strip_records("p1", n=200)
    splits = assign_spatial_blocks(records, FRACTIONS, buffer_m=TILE_M)
    kept = sum(len(v) for v in splits.values())
    assert kept < len(records)  # buffer strips removed something
    # every record is either kept once or dropped — never duplicated
    ids = [r.tile_id for v in splits.values() for r in v]
    assert len(ids) == len(set(ids))


def test_augmented_only_in_train_region():
    base = strip_records("p1", n=200)
    aug = strip_records("p1", n=200, augmented=True, y0=10.0)
    splits = assign_spatial_blocks(base + aug, FRACTIONS, buffer_m=TILE_M)
    assert any(r.augmented for r in splits["train"])
    assert not any(r.augmented for r in splits["val"])
    assert not any(r.augmented for r in splits["test"])
    # aug tiles in val/test regions were dropped, not reassigned to train:
    # layout is VAL | TRAIN | TEST, so every augmented train tile must sit
    # strictly between the val region and the test region.
    val_u_max = max(u_of(r) for r in splits["val"])
    test_u_min = min(u_of(r) for r in splits["test"])
    for r in splits["train"]:
        if r.augmented:
            assert val_u_max < u_of(r) < test_u_min


def test_multiple_polygons_pool_into_shared_splits():
    p1 = strip_records("p1", n=100)
    p2 = strip_records("p2", n=100, theta_deg=-33.0)
    splits = assign_spatial_blocks(p1 + p2, FRACTIONS, buffer_m=TILE_M)
    for name in ("train", "val", "test"):
        assert {r.polygon for r in splits[name]} == {"p1", "p2"}


def test_tiny_polygon_tolerated_when_pooled():
    """A 12-tile polygon (like polygon5) may lose val/test to the buffer — fine
    as long as the pooled split still has val/test from other polygons."""
    big = strip_records("p1", n=200)
    tiny = strip_records("p5", n=12)
    splits = assign_spatial_blocks(big + tiny, FRACTIONS, buffer_m=TILE_M)
    assert splits["val"] and splits["test"]
    assert any(r.polygon == "p5" for v in splits.values() for r in v)


def test_empty_val_or_test_raises():
    # 3 tiles cannot fill three regions once the buffer eats the boundaries
    with pytest.raises(ValueError, match="empty"):
        assign_spatial_blocks(strip_records("p1", n=3), FRACTIONS, buffer_m=TILE_M)


def test_aug_only_polygon_rejected():
    with pytest.raises(ValueError, match="no base"):
        assign_spatial_blocks(
            strip_records("p1", n=10, augmented=True), FRACTIONS, buffer_m=TILE_M
        )
