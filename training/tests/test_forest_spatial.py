"""seabed_forest.spatial: mask-aware guided filter + posterior regularization."""

import numpy as np

from seabed_forest.spatial import guided_filter, regularize_posterior


def test_guided_filter_smooths_uniform_region():
    # Flat guide + noisy source -> output ~ local mean (denoised).
    rng = np.random.default_rng(0)
    guide = np.zeros((20, 20), np.float64)
    src = 0.5 + rng.normal(0, 0.2, (20, 20))
    mask = np.ones((20, 20), bool)
    out = guided_filter(guide, src, radius=3, eps=1e-3, mask=mask)
    # variance is reduced relative to the noisy input
    assert out.std() < src.std()
    assert out.shape == src.shape


def test_guided_filter_preserves_guide_edge():
    # A sharp step in the guide should be preserved in the output (edge-aware),
    # unlike a plain blur which bleeds across it.
    guide = np.zeros((20, 20), np.float64)
    guide[:, 10:] = 1.0
    src = guide.copy()                      # source follows the same edge
    mask = np.ones((20, 20), bool)
    out = guided_filter(guide, src, radius=4, eps=1e-6, mask=mask)
    # the jump across the edge (col 9 -> col 10) stays close to 1.0
    jump = out[:, 10].mean() - out[:, 9].mean()
    assert jump > 0.8


def test_mask_aware_ignores_uncovered_pixels():
    # Uncovered region carries a wild value but must not bleed into covered output.
    guide = np.zeros((10, 10), np.float64)
    src = np.full((10, 10), 0.5)
    src[0:2, :] = 1000.0                     # garbage in the uncovered band
    mask = np.ones((10, 10), bool)
    mask[0:2, :] = False                     # those rows are uncovered
    out = guided_filter(guide, src, radius=2, eps=1e-3, mask=mask)
    # covered pixels far from the garbage stay ~0.5, not pulled toward 1000
    assert abs(out[6, 5] - 0.5) < 0.1
    # row 2: its radius-2 window straddles the uncovered garbage rows (0-1),
    # so this is the real test that mask-aware box means exclude them.
    assert abs(out[2, 5] - 0.5) < 0.1
    assert np.isfinite(out).all()


def test_regularize_posterior_renormalizes_and_cleans_singleton():
    # 3-class posterior, mostly class 0, a single class-2 spike -> smoothed away.
    C, H, W = 3, 15, 15
    prob = np.zeros((C, H, W), np.float64)
    prob[0] = 0.8; prob[1] = 0.1; prob[2] = 0.1
    prob[:, 7, 7] = [0.0, 0.0, 1.0]          # lone class-2 spike
    guide = np.zeros((H, W), np.float64)
    mask = np.ones((H, W), bool)
    reg = regularize_posterior(prob, guide, mask, radius=3, eps=1e-3)
    assert reg.shape == (C, H, W)
    # columns sum to 1 over covered pixels
    sums = reg.sum(axis=0)
    assert np.allclose(sums[mask], 1.0, atol=1e-6)
    # the lone spike no longer wins argmax (neighbours are class 0)
    assert reg[:, 7, 7].argmax() == 0


def test_regularize_posterior_leaves_uncovered_as_zero():
    C, H, W = 3, 8, 8
    prob = np.full((C, H, W), 1.0 / 3.0)
    guide = np.zeros((H, W), np.float64)
    mask = np.ones((H, W), bool)
    mask[:, 0] = False                       # first column uncovered
    reg = regularize_posterior(prob, guide, mask, radius=2, eps=1e-3)
    assert np.allclose(reg[:, ~mask].sum(axis=0), 0.0)   # uncovered -> all-zero column
    assert np.isfinite(reg).all()
