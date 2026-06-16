"""Evaluation metrics over non-ignored pixels (numpy, torch-free).

Matches the reference paper's suite: per-class + macro Dice, overall accuracy,
Cohen's kappa, producer's accuracy (recall) and user's accuracy (precision),
all derived from one confusion matrix. Rows = reference (true), cols = predicted.
"""

from __future__ import annotations

import numpy as np

from .data import IGNORE_INDEX


def confusion_matrix(
    pred: np.ndarray, target: np.ndarray, num_classes: int
) -> np.ndarray:
    """(C, C) int64 matrix over pixels where target != IGNORE_INDEX."""
    valid = target != IGNORE_INDEX
    t = target[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    return np.bincount(
        t * num_classes + p, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def _safe_div(num: np.ndarray, denom: np.ndarray) -> np.ndarray:
    out = np.full_like(num, np.nan, dtype=np.float64)
    np.divide(num, denom, out=out, where=denom > 0)
    return out


def dice_per_class(cm: np.ndarray) -> np.ndarray:
    diag = np.diag(cm).astype(np.float64)
    return _safe_div(2.0 * diag, cm.sum(axis=0) + cm.sum(axis=1))


def producers_accuracy(cm: np.ndarray) -> np.ndarray:
    """Recall per true class (PAcc)."""
    return _safe_div(np.diag(cm).astype(np.float64), cm.sum(axis=1))


def users_accuracy(cm: np.ndarray) -> np.ndarray:
    """Precision per predicted class (UAcc)."""
    return _safe_div(np.diag(cm).astype(np.float64), cm.sum(axis=0))


def overall_accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total) if total else float("nan")


def cohens_kappa(cm: np.ndarray) -> float:
    total = cm.sum()
    if not total:
        return float("nan")
    po = np.trace(cm) / total
    pe = float((cm.sum(axis=1) * cm.sum(axis=0)).sum()) / (total * total)
    if pe == 1.0:
        return float("nan")
    return float((po - pe) / (1.0 - pe))


def metrics_report(cm: np.ndarray, class_names: list[str]) -> dict:
    """All metrics as one JSON-serializable dict."""
    dice = dice_per_class(cm)
    pacc = producers_accuracy(cm)
    uacc = users_accuracy(cm)
    return {
        "overall_accuracy": overall_accuracy(cm),
        "cohens_kappa": cohens_kappa(cm),
        "macro_dice": float(np.nanmean(dice)),
        "per_class": {
            name: {
                "dice": float(dice[i]),
                "producers_accuracy": float(pacc[i]),
                "users_accuracy": float(uacc[i]),
                "support_px": int(cm.sum(axis=1)[i]),
            }
            for i, name in enumerate(class_names)
        },
        "confusion_matrix": cm.tolist(),
    }
