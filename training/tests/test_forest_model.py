"""seabed_forest.model: build/fit both estimators, channel-aligned probabilities, IO."""

import numpy as np

from seabed_forest.config import ForestConfig
from seabed_forest.model import (
    build_estimator, fit_estimator, load_model, predict_proba_channels, save_model,
)

NUM_CLASSES = 3


def _toy_data(seed=0):
    rng = np.random.default_rng(seed)
    # 3 separable blobs in 3-D so trees learn something deterministic.
    X = np.vstack([rng.normal(c, 0.2, size=(80, 3)) for c in (0.0, 1.0, 2.0)]).astype(np.float32)
    y = np.repeat([0, 1, 2], 80).astype(np.int64)
    return X, y


def test_both_estimators_fit_and_predict():
    X, y = _toy_data()
    fc = ForestConfig()
    for kind in fc.models:
        est = fit_estimator(build_estimator(kind, fc, NUM_CLASSES), kind, X, y)
        pred = est.predict(X)
        assert pred.shape == (X.shape[0],)
        assert (pred == y).mean() > 0.95


def test_predict_proba_channels_full_width_when_class_absent():
    X, y = _toy_data()
    # Drop class 2 entirely from training -> estimator.classes_ == [0, 1].
    mask = y != 2
    est = fit_estimator(build_estimator("random_forest", ForestConfig(), NUM_CLASSES),
                        "random_forest", X[mask], y[mask])
    proba = predict_proba_channels(est, X, NUM_CLASSES)
    assert proba.shape == (X.shape[0], NUM_CLASSES)
    assert np.allclose(proba[:, 2], 0.0)            # absent channel filled with zeros
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_save_load_roundtrip(tmp_path):
    X, y = _toy_data()
    est = fit_estimator(build_estimator("random_forest", ForestConfig(), NUM_CLASSES),
                        "random_forest", X, y)
    path = tmp_path / "model_random_forest.joblib"
    save_model(est, path)
    assert np.array_equal(load_model(path).predict(X), est.predict(X))


def test_seed_determinism():
    X, y = _toy_data()
    fc = ForestConfig()
    a = fit_estimator(build_estimator("random_forest", fc, NUM_CLASSES), "random_forest", X, y)
    b = fit_estimator(build_estimator("random_forest", fc, NUM_CLASSES), "random_forest", X, y)
    assert np.array_equal(a.predict(X), b.predict(X))
