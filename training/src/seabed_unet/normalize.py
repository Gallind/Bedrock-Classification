"""Per-band [0,1] normalization with robust percentile clipping.

Each band has its own mode (cfg.normalization.band_modes):
- per_polygon: the survey self-normalizes from its own feature pixels. No label
  information is used, so this is legitimate at inference time (a new survey
  normalizes itself). Required for backscatter, whose units differ per survey
  (polygon1 = JPEG grayscale 0-255, polygons 3/4/5 = real dB).
- global: one (lo, hi) from TRAIN-split base-tile pixels only, applied
  everywhere. Preserves absolute physical values across surveys — essential for
  bathymetry, because shallow_rock is *defined* by depth and per-polygon scaling
  would erase the very feature that separates it from rock.

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
    train_features: list[np.ndarray],
    bands: list[str],
    band_modes: dict[str, str],
    nodata: float,
    clip_percentiles: tuple[float, float],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Stats dict: per_polygon bands under each polygon key, global bands under GLOBAL_KEY.

    ``features_by_polygon``: base-tile features of ALL polygons (feature-only —
    no labels — so including val/test surveys leaks nothing).
    ``train_features``: base-tile features of the TRAIN split only; global stats
    must never be fit on val/test pixels.
    """
    per_poly = [(i, b) for i, b in enumerate(bands) if band_modes[b] == "per_polygon"]
    global_ = [(i, b) for i, b in enumerate(bands) if band_modes[b] == "global"]
    unknown = [b for b in bands if band_modes[b] not in ("per_polygon", "global")]
    if unknown:
        raise ValueError(f"unknown normalization mode for band(s) {unknown}")

    stats: dict[str, dict[str, tuple[float, float]]] = {}
    if per_poly:
        for poly, arrays in features_by_polygon.items():
            stats[poly] = {
                band: compute_band_stats(arrays, i, nodata, clip_percentiles)
                for i, band in per_poly
            }
    if global_:
        if not train_features:
            raise ValueError("global-mode bands require train_features")
        stats[GLOBAL_KEY] = {
            band: compute_band_stats(train_features, i, nodata, clip_percentiles)
            for i, band in global_
        }
    return stats


def apply_stats(
    features: np.ndarray,
    polygon: str,
    bands: list[str],
    stats: dict[str, dict[str, tuple[float, float]]],
    nodata: float,
    band_modes: dict[str, str],
) -> np.ndarray:
    """Normalize (B,H,W) features to [0,1]; invalid pixels become 0.0.

    Invalid (nodata/NaN) pixels are excluded from the loss anyway — they just
    need a finite value so the network input is well-defined.
    """
    out = np.empty_like(features, dtype=np.float32)
    for i, band in enumerate(bands):
        key = GLOBAL_KEY if band_modes[band] == "global" else polygon
        if key not in stats or band not in stats[key]:
            raise KeyError(f"no normalization stats for band {band!r} of {polygon!r}")
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
