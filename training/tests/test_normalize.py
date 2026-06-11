"""normalize.py: stats computation modes, application, JSON roundtrip."""

import numpy as np
import pytest

from seabed_unet.normalize import (
    GLOBAL_KEY,
    apply_stats,
    compute_stats,
    load_stats,
    save_stats,
)

NODATA = -9999.0
BANDS = ["bathymetry", "slope"]


def make_features(lo: float, hi: float) -> np.ndarray:
    """(2, 4, 4) array with each band linearly spanning [lo, hi]."""
    band = np.linspace(lo, hi, 16, dtype=np.float32).reshape(4, 4)
    return np.stack([band, band + 0.0])


def test_per_polygon_mode_self_normalizes_each_survey():
    grouped = {"polygon1": [make_features(0, 255)], "polygon3": [make_features(-50, 0)]}
    stats = compute_stats(grouped, BANDS, "per_polygon", NODATA, (0.0, 100.0))
    for poly in grouped:
        normalized = apply_stats(grouped[poly][0], poly, BANDS, stats, NODATA)
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)


def test_global_mode_uses_train_polygons_only():
    grouped = {"train_poly": [make_features(0, 100)], "test_poly": [make_features(0, 200)]}
    stats = compute_stats(
        grouped, BANDS, "global", NODATA, (0.0, 100.0), train_polygons=["train_poly"]
    )
    assert set(stats) == {GLOBAL_KEY}
    assert stats[GLOBAL_KEY]["bathymetry"] == (0.0, 100.0)  # test_poly never contributed
    # test_poly values above the train max are clipped to 1.0
    normalized = apply_stats(grouped["test_poly"][0], "test_poly", BANDS, stats, NODATA)
    assert normalized.max() == pytest.approx(1.0)
    assert (normalized == 1.0).sum() > 2  # several clipped pixels, not just the endpoint


def test_global_mode_requires_train_polygons():
    with pytest.raises(ValueError, match="train_polygons"):
        compute_stats({"p": [make_features(0, 1)]}, BANDS, "global", NODATA, (0, 100))


def test_nodata_excluded_from_stats_and_zeroed_in_output():
    features = make_features(0, 100)
    features[0, 0, :] = NODATA
    features[1, 1, 0] = np.nan
    stats = compute_stats({"p": [features]}, BANDS, "per_polygon", NODATA, (0.0, 100.0))
    lo, hi = stats["p"]["bathymetry"]
    assert lo > NODATA  # nodata never contaminates the percentile
    normalized = apply_stats(features, "p", BANDS, stats, NODATA)
    assert (normalized[0, 0, :] == 0.0).all()
    assert normalized[1, 1, 0] == 0.0
    assert np.isfinite(normalized).all()


def test_constant_band_does_not_divide_by_zero():
    features = np.full((2, 4, 4), 5.0, dtype=np.float32)
    stats = compute_stats({"p": [features]}, BANDS, "per_polygon", NODATA, (2.0, 98.0))
    normalized = apply_stats(features, "p", BANDS, stats, NODATA)
    assert np.isfinite(normalized).all()


def test_clip_percentiles_trim_outliers():
    # 100 pixels per band so a single outlier (1%) falls beyond the 98th percentile
    band = np.linspace(0, 100, 100, dtype=np.float32).reshape(10, 10)
    features = np.stack([band, band.copy()])
    features[0, 9, 9] = 1e6  # single outlier
    stats = compute_stats({"p": [features]}, BANDS, "per_polygon", NODATA, (2.0, 98.0))
    _, hi = stats["p"]["bathymetry"]
    assert hi < 1000.0


def test_stats_json_roundtrip(tmp_path):
    grouped = {"polygon1": [make_features(0, 255)]}
    stats = compute_stats(grouped, BANDS, "per_polygon", NODATA, (2.0, 98.0))
    path = tmp_path / "runs" / "stats.json"
    save_stats(stats, path)
    assert load_stats(path) == stats


def test_unknown_polygon_without_global_raises():
    stats = {"polygon1": {"bathymetry": (0.0, 1.0), "slope": (0.0, 1.0)}}
    with pytest.raises(KeyError, match="polygon9"):
        apply_stats(make_features(0, 1), "polygon9", BANDS, stats, NODATA)
