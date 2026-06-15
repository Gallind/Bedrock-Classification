"""Edge-aware spatial regularization of the per-pixel posterior (pure numpy/scipy).

A guided filter (He, Sun & Tang 2013) smooths each class-probability channel while
preserving edges in a guide image (a feature band, e.g. bathymetry = depth boundaries),
so geological boundaries survive while salt-and-pepper is removed. All box means are
mask-aware (normalized by the coverage mask) so uncovered/nodata pixels never bleed in.
Implemented with scipy.ndimage.uniform_filter only — no new dependencies, no CRF.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter


def _box(x: np.ndarray, radius: int) -> np.ndarray:
    """Box-sum-like mean over a (2*radius+1) window (constant/zero padding at borders)."""
    return uniform_filter(x, size=2 * radius + 1, mode="constant")


def _box_mean(x: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    """Mask-aware window mean: average only over covered pixels in each window."""
    num = _box(x * mask, radius)
    den = _box(mask, radius)
    out = np.zeros_like(x)
    np.divide(num, den, out=out, where=den > 0)
    return out


def guided_filter(
    guide: np.ndarray, src: np.ndarray, radius: int, eps: float, mask: np.ndarray
) -> np.ndarray:
    """Single-channel guided filter of ``src`` under ``guide`` (both 2-D, float).

    Mask-aware: window statistics use covered pixels only. Returns a float array the
    shape of ``src``; values at uncovered pixels are not meaningful (callers re-mask).
    """
    g = guide.astype(np.float64)
    p = src.astype(np.float64)
    m = mask.astype(np.float64)

    mean_g = _box_mean(g, m, radius)
    mean_p = _box_mean(p, m, radius)
    corr_g = _box_mean(g * g, m, radius)
    corr_gp = _box_mean(g * p, m, radius)
    var_g = corr_g - mean_g * mean_g
    cov_gp = corr_gp - mean_g * mean_p

    a = cov_gp / (var_g + eps)
    b = mean_p - a * mean_g
    mean_a = _box_mean(a, m, radius)
    mean_b = _box_mean(b, m, radius)
    return mean_a * g + mean_b


def regularize_posterior(
    prob: np.ndarray, guide: np.ndarray, mask: np.ndarray, radius: int, eps: float
) -> np.ndarray:
    """Edge-aware-smooth each class channel of ``prob`` (C,H,W) under ``guide`` (H,W),
    then clip >= 0 and renormalize across classes over covered pixels.

    Uncovered pixels (mask False) are forced to an all-zero column so a downstream
    argmax + coverage check behaves exactly like the unregularized map.
    """
    c = prob.shape[0]
    out = np.stack([
        guided_filter(guide, prob[k], radius=radius, eps=eps, mask=mask) for k in range(c)
    ])
    np.clip(out, 0.0, None, out=out)
    out *= mask[np.newaxis, :, :]
    total = out.sum(axis=0)
    safe = total > 0
    out[:, safe] /= total[safe]
    return out
