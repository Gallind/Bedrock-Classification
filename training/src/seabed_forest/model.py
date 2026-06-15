"""scikit-learn estimator wrapper for the per-pixel baseline.

Targets are model-CHANNEL indices (0..num_classes-1), so sklearn's classes_ are the
channel indices and predict_proba columns are already in channel order — except when a
class is absent from a training fold, which predict_proba_channels repairs by inserting
zero columns. RF carries class_weight='balanced'; HGB has no class_weight, so we pass
balanced sample weights at fit time.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.utils.class_weight import compute_sample_weight

from .config import ForestConfig


def build_estimator(kind: str, forest: ForestConfig, num_classes: int):
    if kind == "random_forest":
        p = forest.random_forest
        return RandomForestClassifier(
            n_estimators=p.n_estimators, max_depth=p.max_depth,
            min_samples_leaf=p.min_samples_leaf, n_jobs=p.n_jobs,
            class_weight=p.class_weight, random_state=forest.seed,
        )
    if kind == "hist_gradient_boosting":
        p = forest.hist_gradient_boosting
        return HistGradientBoostingClassifier(
            learning_rate=p.learning_rate, max_iter=p.max_iter,
            max_leaf_nodes=p.max_leaf_nodes, l2_regularization=p.l2_regularization,
            early_stopping=p.early_stopping, random_state=forest.seed,
        )
    raise ValueError(f"unknown estimator kind {kind!r}")


def fit_estimator(estimator, kind: str, X: np.ndarray, y: np.ndarray):
    """Fit; HGB gets balanced sample weights (it has no class_weight param)."""
    if kind == "hist_gradient_boosting":
        sw = compute_sample_weight("balanced", y)
        estimator.fit(X, y, sample_weight=sw)
    else:
        estimator.fit(X, y)
    return estimator


def predict_proba_channels(estimator, X: np.ndarray, num_classes: int) -> np.ndarray:
    """(N, num_classes) probabilities in channel order; absent classes -> zero columns."""
    proba = estimator.predict_proba(X)
    out = np.zeros((X.shape[0], num_classes), dtype=np.float64)
    for col, cls in enumerate(estimator.classes_):
        out[:, int(cls)] = proba[:, col]
    return out


def feature_importance(
    estimator, kind: str, X: np.ndarray, y: np.ndarray, bands: list[str], seed: int,
    sample: int = 50_000,
) -> dict[str, float]:
    """{band: importance}. RF: impurity importances. HGB: permutation importance on a
    subsample (permutation is model-agnostic but O(n_features * n_samples))."""
    if kind == "random_forest":
        values = np.asarray(estimator.feature_importances_, dtype=float)
    else:
        rng = np.random.default_rng(seed)
        idx = (rng.choice(X.shape[0], size=sample, replace=False)
               if X.shape[0] > sample else np.arange(X.shape[0]))
        result = permutation_importance(
            estimator, X[idx], y[idx], n_repeats=5, random_state=seed, n_jobs=-1,
        )
        values = np.asarray(result.importances_mean, dtype=float)
    return {band: float(v) for band, v in zip(bands, values)}


def save_model(estimator, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, path)


def load_model(path: Path):
    return joblib.load(path)
