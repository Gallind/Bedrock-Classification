"""normalize.py: per-band modes, train-only global stats, JSON roundtrip."""

import numpy as np
import pytest

from seabed_unet.normalize import (
    GLOBAL_KEY,
    apply_stats,
    compute_band_stats,
    compute_stats,
    load_stats,
    save_stats,
)

NODATA = -9999.0
BANDS = ["bathymetry", "slope"]
PER_POLY = {b: "per_polygon" for b in BANDS}
GLOBAL = {b: "global" for b in BANDS}


def make_features(lo: float, hi: float, n_bands: int = 2) -> np.ndarray:
    """(n_bands, 4, 4) array with each band linearly spanning [lo, hi]."""
    band = np.linspace(lo, hi, 16, dtype=np.float32).reshape(4, 4)
    return np.stack([band.copy() for _ in range(n_bands)])


def test_per_polygon_mode_self_normalizes_each_survey():
    grouped = {"polygon1": [make_features(0, 255)], "polygon3": [make_features(-50, 0)]}
    stats = compute_stats(grouped, [], BANDS, PER_POLY, NODATA, (0.0, 100.0))
    for poly in grouped:
        normalized = apply_stats(grouped[poly][0], poly, BANDS, stats, NODATA, PER_POLY)
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)


def test_global_mode_uses_train_features_only():
    train = make_features(0, 100)
    test = make_features(0, 200)
    grouped = {"train_poly": [train], "test_poly": [test]}
    stats = compute_stats(grouped, [train], BANDS, GLOBAL, NODATA, (0.0, 100.0))
    assert set(stats) == {GLOBAL_KEY}
    assert stats[GLOBAL_KEY]["bathymetry"] == (0.0, 100.0)  # test_poly never contributed
    normalized = apply_stats(test, "test_poly", BANDS, stats, NODATA, GLOBAL)
    assert normalized.max() == pytest.approx(1.0)
    assert (normalized == 1.0).sum() > 2  # values above train range clip to 1


def test_global_mode_requires_train_features():
    with pytest.raises(ValueError, match="train_features"):
        compute_stats({"p": [make_features(0, 1)]}, [], BANDS, GLOBAL, NODATA, (0, 100))


def test_mixed_modes_global_depth_preserves_cross_survey_ordering():
    """The motivating case: global bathymetry keeps a deep survey's pixels
    uniformly 'deeper' than a shallow survey's after normalization, while the
    per-polygon band still self-normalizes."""
    bands = ["backscatter", "bathymetry"]
    modes = {"backscatter": "per_polygon", "bathymetry": "global"}
    deep = np.stack([make_features(0, 255)[0], make_features(-30, -20)[0]])
    shallow = np.stack([make_features(-50, 0)[0], make_features(-10, 0)[0]])
    grouped = {"deep_poly": [deep], "shallow_poly": [shallow]}
    stats = compute_stats(grouped, [deep, shallow], bands, modes, NODATA, (0.0, 100.0))

    n_deep = apply_stats(deep, "deep_poly", bands, stats, NODATA, modes)
    n_shallow = apply_stats(shallow, "shallow_poly", bands, stats, NODATA, modes)
    # bathymetry: every deep pixel stays below every shallow pixel
    assert n_deep[1].max() < n_shallow[1].min()
    # backscatter: both surveys span [0,1] despite different units
    for n in (n_deep, n_shallow):
        assert n[0].min() == pytest.approx(0.0) and n[0].max() == pytest.approx(1.0)


def test_unknown_band_mode_rejected():
    with pytest.raises(ValueError, match="unknown normalization mode"):
        compute_stats(
            {"p": [make_features(0, 1)]}, [], BANDS,
            {"bathymetry": "zscore", "slope": "global"}, NODATA, (0, 100),
        )


def test_nodata_excluded_from_stats_and_zeroed_in_output():
    features = make_features(0, 100)
    features[0, 0, :] = NODATA
    features[1, 1, 0] = np.nan
    stats = compute_stats({"p": [features]}, [], BANDS, PER_POLY, NODATA, (0.0, 100.0))
    lo, _ = stats["p"]["bathymetry"]
    assert lo > NODATA  # nodata never contaminates the percentile
    normalized = apply_stats(features, "p", BANDS, stats, NODATA, PER_POLY)
    assert (normalized[0, 0, :] == 0.0).all()
    assert normalized[1, 1, 0] == 0.0
    assert np.isfinite(normalized).all()


def test_constant_band_does_not_divide_by_zero():
    features = np.full((2, 4, 4), 5.0, dtype=np.float32)
    stats = compute_stats({"p": [features]}, [], BANDS, PER_POLY, NODATA, (2.0, 98.0))
    normalized = apply_stats(features, "p", BANDS, stats, NODATA, PER_POLY)
    assert np.isfinite(normalized).all()


def test_clip_percentiles_trim_outliers():
    # 100 pixels per band so a single outlier (1%) falls beyond the 98th percentile
    band = np.linspace(0, 100, 100, dtype=np.float32).reshape(10, 10)
    features = np.stack([band, band.copy()])
    features[0, 9, 9] = 1e6  # single outlier
    _, hi = compute_band_stats([features], 0, NODATA, (2.0, 98.0))
    assert hi < 1000.0


def test_stats_json_roundtrip(tmp_path):
    grouped = {"polygon1": [make_features(0, 255)]}
    stats = compute_stats(
        grouped, [grouped["polygon1"][0]], BANDS,
        {"bathymetry": "global", "slope": "per_polygon"}, NODATA, (2.0, 98.0),
    )
    assert GLOBAL_KEY in stats and "polygon1" in stats
    path = tmp_path / "runs" / "stats.json"
    save_stats(stats, path)
    assert load_stats(path) == stats


def test_unknown_polygon_for_per_polygon_band_raises():
    stats = {"polygon1": {"bathymetry": (0.0, 1.0), "slope": (0.0, 1.0)}}
    with pytest.raises(KeyError, match="polygon9"):
        apply_stats(make_features(0, 1), "polygon9", BANDS, stats, NODATA, PER_POLY)
