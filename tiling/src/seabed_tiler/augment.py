# tiling/src/seabed_tiler/augment.py
"""Training-time D4 (dihedral group) augmentation for tile pairs.

This module is numpy-only on purpose: the future training project can import or
vendor it without pulling in any geospatial or deep-learning dependency.

The public API transforms the (features, label) pair as ONE atomic unit. There is
deliberately no function that transforms a single array: features and label must
never be transformed independently, exactly like the raw survey bundles
(shp/prj/jgw/jpg) must never be edited file-by-file. Transforming one half of a
pair silently destroys feature-label co-registration without raising any error.

Only the 8 rigid ops of the dihedral group D4 are offered. They are exact index
permutations (zero interpolation) and geophysically valid for MBES data: slope
magnitude is rotation-invariant and bathymetry/backscatter have no preferred
direction at tile scale. Photometric transforms (brightness, contrast, noise,
depth offsets, scaling) are prohibited -- pixel values are physical measurements.

See docs/DATA_AUGMENTATION.md for the full dataset-consumer contract.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# The 8 ops of D4. rot* are counter-clockwise (numpy rot90 convention) in array
# index space; fliph mirrors left-right (width axis), flipv mirrors top-bottom.
D4_OPS: tuple[str, ...] = (
    "identity",
    "rot90",
    "rot180",
    "rot270",
    "fliph",
    "flipv",
    "transpose",
    "anti_transpose",
)

# Every D4 op is its own inverse except the quarter turns.
_INVERSE: dict[str, str] = {
    "identity": "identity",
    "rot90": "rot270",
    "rot180": "rot180",
    "rot270": "rot90",
    "fliph": "fliph",
    "flipv": "flipv",
    "transpose": "transpose",
    "anti_transpose": "anti_transpose",
}


def _apply(arr: np.ndarray, op: str, h_axis: int, w_axis: int) -> np.ndarray:
    if op == "identity":
        return arr
    if op == "rot90":
        return np.rot90(arr, k=1, axes=(h_axis, w_axis))
    if op == "rot180":
        return np.rot90(arr, k=2, axes=(h_axis, w_axis))
    if op == "rot270":
        return np.rot90(arr, k=3, axes=(h_axis, w_axis))
    if op == "fliph":
        return np.flip(arr, axis=w_axis)
    if op == "flipv":
        return np.flip(arr, axis=h_axis)
    if op == "transpose":
        return np.swapaxes(arr, h_axis, w_axis)
    if op == "anti_transpose":
        return np.rot90(np.swapaxes(arr, h_axis, w_axis), k=2, axes=(h_axis, w_axis))
    raise ValueError(f"unknown D4 op: {op!r} (valid: {', '.join(D4_OPS)})")


def augment_pair(
    features: np.ndarray, label: np.ndarray, op: str
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one D4 op to a co-registered (features, label) tile pair.

    features: (B, H, W) array, label: (H, W) array with H == W (square tiles).
    Returns new C-contiguous arrays; inputs are not modified.
    """
    if op not in D4_OPS:
        raise ValueError(f"unknown D4 op: {op!r} (valid: {', '.join(D4_OPS)})")
    if features.ndim != 3:
        raise ValueError(f"features must be (B, H, W), got shape {features.shape}")
    if label.ndim != 2:
        raise ValueError(f"label must be (H, W), got shape {label.shape}")
    if features.shape[1:] != label.shape:
        raise ValueError(
            f"features spatial shape {features.shape[1:]} != label shape {label.shape}"
        )
    if label.shape[0] != label.shape[1]:
        raise ValueError(f"tiles must be square, got shape {label.shape}")

    out_features = np.ascontiguousarray(_apply(features, op, h_axis=1, w_axis=2))
    out_label = np.ascontiguousarray(_apply(label, op, h_axis=0, w_axis=1))
    return out_features, out_label


def inverse_op(op: str) -> str:
    """Return the D4 op that undoes ``op``."""
    if op not in _INVERSE:
        raise ValueError(f"unknown D4 op: {op!r} (valid: {', '.join(D4_OPS)})")
    return _INVERSE[op]


def random_d4(rng: np.random.Generator) -> str:
    """Draw a uniformly random D4 op. Pass a seeded Generator for reproducibility."""
    return D4_OPS[int(rng.integers(0, len(D4_OPS)))]
