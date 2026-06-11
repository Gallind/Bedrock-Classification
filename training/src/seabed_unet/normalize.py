"""Per-band [0,1] normalization with robust percentile clipping.

Two modes (cfg.normalization.mode):
- per_polygon: each survey is self-normalized from its own feature pixels. No
  label information is used, so this is legitimate at inference time (a new
  survey normalizes itself) — and it is what bridges the backscatter domain
  shift (polygon1 = JPEG grayscale 0-255, polygons 3/4/5 = real dB).
- global: stats computed from TRAIN polygons only and applied everywhere
  (paper-faithful; normalization params must never be fit on val/test).

Stats are computed from base (_rot) tiles only, so they do not depend on how
many augmentation passes exist, and are saved as JSON next to the checkpoint
for reuse at inference time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

GLOBAL_KEY = "__global__"


def compute_band_stats(
    arrays: list[np.ndarray],
    band_idx: int,
    nodata: float,
    clip_percentiles: tuple[float, float],
) -> tuple[float, float]:
    """(lo, hi) percentiles of valid pixels of one band across feature arrays (B,H,W)."""
    samples = []
    for arr in arrays:
        band = arr[band_idx]
        valid = band[(band != nodata) & ~np.isnan(band)]
        if valid.size:
            samples.append(valid)
    if not samples:
        raise ValueError(f"no valid pixels for band index {band_idx}")
    allpx = np.concatenate(samples)
    lo, hi = np.percentile(allpx, clip_percentiles)
    if hi <= lo:  # constant band; avoid division by zero downstream
        hi = lo + 1.0
    return float(lo), float(hi)


def compute_stats(
    features_by_polygon: dict[str, list[np.ndarray]],
    bands: list[str],
    mode: str,
    nodata: float,
    clip_percentiles: tuple[float, float],
    train_polygons: list[str] | None = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Stats dict keyed by polygon (per_polygon) or GLOBAL_KEY (global).

    In global mode only ``train_polygons``' pixels contribute.
    """
    if mode == "per_polygon":
        return {
            poly: {
                band: compute_band_stats(arrays, i, nodata, clip_percentiles)
                for i, band in enumerate(bands)
            }
            for poly, arrays in features_by_polygon.items()
        }
    if mode == "global":
        if not train_polygons:
            raise ValueError("global mode requires train_polygons")
        train_arrays = [
            arr for poly in train_polygons for arr in features_by_polygon[poly]
        ]
        return {
            GLOBAL_KEY: {
                band: compute_band_stats(train_arrays, i, nodata, clip_percentiles)
                for i, band in enumerate(bands)
            }
        }
    raise ValueError(f"unknown normalization mode: {mode!r}")


def apply_stats(
    features: np.ndarray,
    polygon: str,
    bands: list[str],
    stats: dict[str, dict[str, tuple[float, float]]],
    nodata: float,
) -> np.ndarray:
    """Normalize (B,H,W) features to [0,1]; invalid pixels become 0.0.

    Invalid (nodata/NaN) pixels are excluded from the loss anyway — they just
    need a finite value so the network input is well-defined.
    """
    key = polygon if polygon in stats else GLOBAL_KEY
    if key not in stats:
        raise KeyError(f"no normalization stats for polygon {polygon!r}")
    out = np.empty_like(features, dtype=np.float32)
    for i, band in enumerate(bands):
        lo, hi = stats[key][band]
        src = features[i]
        invalid = (src == nodata) | np.isnan(src)
        scaled = (np.clip(src, lo, hi) - lo) / (hi - lo)
        scaled[invalid] = 0.0
        out[i] = scaled
    return out


def save_stats(stats: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2, sort_keys=True))


def load_stats(path: Path) -> dict[str, dict[str, tuple[float, float]]]:
    raw = json.loads(Path(path).read_text())
    return {
        key: {band: (float(lo), float(hi)) for band, (lo, hi) in bands.items()}
        for key, bands in raw.items()
    }
