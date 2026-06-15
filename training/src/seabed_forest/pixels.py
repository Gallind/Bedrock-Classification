"""Turn TileRecords into a flat per-pixel table (X, y, world coords).

Per-pixel, context-free: each valid pixel is one (band-values -> class) sample.
- Normalize with the U-Net's apply_stats (same per-band modes).
- Encode labels with the U-Net's encode_target: class_id -> channel index, with
  IGNORE_INDEX for background (label 0) and any feature-nodata/NaN pixel.
- Drop ignored pixels.
- Optionally dedup train pixels by world coordinate (the _rot tiles overlap 50%, so
  the same ground location appears in several tiles and would otherwise be oversampled).
- Optionally cap pixels per class (stratified subsample) for tractability.
"""

from __future__ import annotations

import numpy as np

from seabed_unet.data import IGNORE_INDEX, TileRecord, encode_target
from seabed_unet.normalize import apply_stats


def _world_coords(transform, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(N, 2) UTM coords of pixel CENTERS for an affine transform (handles rotation)."""
    cc = cols.astype(np.float64) + 0.5
    rr = rows.astype(np.float64) + 0.5
    x = transform.a * cc + transform.b * rr + transform.c
    y = transform.d * cc + transform.e * rr + transform.f
    return np.column_stack([x, y])


def build_pixel_table(
    records: list[TileRecord],
    bands: list[str],
    class_ids: list[int],
    stats: dict,
    nodata: float,
    ignore_label: int,
    band_modes: dict[str, str],
    *,
    dedup: bool = False,
    max_pixels_per_class: int | None = None,
    seed: int = 42,
    round_m: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X (N, B) float32, y (N,) int64 channel indices, coords (N, 2) float64)."""
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    cs: list[np.ndarray] = []
    for r in records:
        norm = apply_stats(r.features, r.polygon, bands, stats, nodata, band_modes)
        target = encode_target(r.label, r.features, class_ids, nodata, ignore_label)
        valid = target != IGNORE_INDEX
        if not valid.any():
            continue
        rows, cols = np.nonzero(valid)
        xs.append(norm[:, rows, cols].T.astype(np.float32))   # (n, B)
        ys.append(target[rows, cols].astype(np.int64))         # (n,)
        cs.append(_world_coords(r.transform, rows, cols))      # (n, 2)

    if not xs:
        return (np.empty((0, len(bands)), np.float32),
                np.empty((0,), np.int64),
                np.empty((0, 2), np.float64))

    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    coords = np.concatenate(cs, axis=0)

    if dedup:
        keys = np.floor(coords / round_m).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        keep.sort()
        X, y, coords = X[keep], y[keep], coords[keep]

    if max_pixels_per_class is not None:
        rng = np.random.default_rng(seed)
        keep_idx: list[np.ndarray] = []
        for cls in np.unique(y):
            idx = np.flatnonzero(y == cls)
            if idx.size > max_pixels_per_class:
                idx = rng.choice(idx, size=max_pixels_per_class, replace=False)
            keep_idx.append(idx)
        keep = np.sort(np.concatenate(keep_idx))
        X, y, coords = X[keep], y[keep], coords[keep]

    return X, y, coords
