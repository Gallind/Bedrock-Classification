"""Two-stage hierarchical classifier for seabed classification.

Stage 1: rock (class 0) vs non-rock (class 1) — binary.
Stage 2: sand vs shallow_rock — only on pixels Stage 1 called non-rock.

Channel indices (matching seabed_unet encode_target with class_ids=[1,2,3]):
  0 = rock, 1 = shallow_rock, 2 = sand
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

ROCK_CH = 0        # channel index for rock
SHALLOW_CH = 1     # channel index for shallow_rock
SAND_CH = 2        # channel index for sand

# Binary label for Stage 1
_S1_ROCK = 0
_S1_NONROCK = 1

# Binary label for Stage 2
_S2_SHALLOW = 0
_S2_SAND = 1


def to_stage1(y: np.ndarray) -> np.ndarray:
    """Map 3-class channel indices to binary rock/non-rock labels."""
    return np.where(y == ROCK_CH, _S1_ROCK, _S1_NONROCK).astype(np.int64)


def to_stage2(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, binary_labels) for non-rock pixels only.
    mask selects pixels where y != ROCK_CH; labels are shallow=0 / sand=1."""
    mask = y != ROCK_CH
    y2 = np.where(y[mask] == SHALLOW_CH, _S2_SHALLOW, _S2_SAND).astype(np.int64)
    return mask, y2


def predict(est1, est2, X: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Return channel-index predictions (N,) combining both stages."""
    pred1 = est1.predict(X)                      # 0=rock, 1=non-rock
    out = np.full(X.shape[0], ROCK_CH, dtype=np.int64)

    nonrock_mask = pred1 == _S1_NONROCK
    if nonrock_mask.any():
        pred2 = est2.predict(X[nonrock_mask])    # 0=shallow_rock, 1=sand
        result = np.where(pred2 == _S2_SHALLOW, SHALLOW_CH, SAND_CH)
        out[nonrock_mask] = result

    return out


def predict_proba(est1, est2, X: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Return (N, num_classes) probability matrix combining both stages."""
    proba1 = _proba2(est1, X, 2)       # (N, 2): [p_rock, p_nonrock]
    out = np.zeros((X.shape[0], num_classes), dtype=np.float64)

    p_rock = proba1[:, _S1_ROCK]
    p_nonrock = proba1[:, _S1_NONROCK]
    out[:, ROCK_CH] = p_rock

    proba2 = _proba2(est2, X, 2)       # (N, 2): [p_shallow, p_sand]
    out[:, SHALLOW_CH] = p_nonrock * proba2[:, _S2_SHALLOW]
    out[:, SAND_CH] = p_nonrock * proba2[:, _S2_SAND]

    return out


def _proba2(est, X: np.ndarray, n: int) -> np.ndarray:
    """predict_proba with missing-class protection."""
    raw = est.predict_proba(X)
    out = np.zeros((X.shape[0], n), dtype=np.float64)
    for col, cls in enumerate(est.classes_):
        out[:, int(cls)] = raw[:, col]
    return out


def save_hierarchical(est1, est2, run_dir: Path, kind: str) -> None:
    joblib.dump(est1, run_dir / f"model_{kind}_stage1.joblib")
    joblib.dump(est2, run_dir / f"model_{kind}_stage2.joblib")


def load_hierarchical(run_dir: Path, kind: str):
    est1 = joblib.load(run_dir / f"model_{kind}_stage1.joblib")
    est2 = joblib.load(run_dir / f"model_{kind}_stage2.joblib")
    return est1, est2
