"""Tile discovery and split assembly (torch-free; the torch Dataset lives in dataset.py).

Binding contract (docs/DATA_AUGMENTATION.md):
- Splits are by WHOLE polygon — tiles overlap 50%, random tile splits leak.
- Augmented (_rotaug) tiles may only ever enter the train split; val/test use
  base (_rot) tiles exclusively. Enforced here with a hard guard, not convention.
- Loss/metrics must ignore every pixel where any feature band is nodata/NaN or
  the label is background (0 = unlabeled, not a class). Encoded here as -1 in
  the target array (IGNORE_INDEX).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from .config import Config

IGNORE_INDEX = -1


@dataclass
class TileRecord:
    tile_id: str
    polygon: str
    augmented: bool          # True iff the tile came from a _rotaug run
    features: np.ndarray     # (B, H, W) float32, raw physical units, cfg.bands order
    label: np.ndarray        # (H, W) uint8 class ids (0=bg, 1=rock, 2=shallow, 3=sand)
    transform: object = None  # affine.Affine of the tile (rotated for _rot runs)
    crs: object = None       # rasterio CRS
    center_x: float = 0.0    # tile center, UTM (manifest column; spatial splits)
    center_y: float = 0.0
    theta_deg: float = 0.0   # grid angle; base tiles carry the polygon's MBR theta


def _read_tile_pair(
    features_path: Path, label_path: Path, bands: list[str]
) -> tuple[np.ndarray, np.ndarray, object, object]:
    """Read the requested bands (by GeoTIFF band description) + the label raster."""
    with rasterio.open(features_path) as src:
        descriptions = list(src.descriptions)
        missing = [b for b in bands if b not in descriptions]
        if missing:
            raise ValueError(
                f"{features_path}: bands {missing} not found in {descriptions}"
            )
        indexes = [descriptions.index(b) + 1 for b in bands]
        features = src.read(indexes).astype(np.float32)
        transform, crs = src.transform, src.crs
    with rasterio.open(label_path) as src:
        label = src.read(1).astype(np.uint8)
    return features, label, transform, crs


def load_run_records(
    run_dir: Path, polygon: str, bands: list[str], base_dir: Path, augmented: bool
) -> list[TileRecord]:
    """Load every tile pair listed in a run's manifest.csv."""
    manifest_path = run_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found — generate the dataset first "
            "(see docs/TRAINING_DATA_SETUP.md)"
        )
    manifest = pd.read_csv(manifest_path)
    records = []
    for row in manifest.itertuples():
        features, label, transform, crs = _read_tile_pair(
            base_dir / row.features_path, base_dir / row.label_path, bands
        )
        records.append(
            TileRecord(
                tile_id=str(row.tile_id),
                polygon=polygon,
                augmented=augmented,
                features=features,
                label=label,
                transform=transform,
                crs=crs,
                center_x=float(row.center_x),
                center_y=float(row.center_y),
                theta_deg=float(row.theta_deg),
            )
        )
    return records


def load_split_records(cfg: Config) -> dict[str, list[TileRecord]]:
    """Assemble {train, val, test} record lists per the binding contract.

    Two modes (cfg.split.mode):
    - polygon: whole-polygon holdout (train/val/test polygon lists).
    - spatial_blocks: all cfg.split.polygons contribute; tiles are assigned to
      contiguous regions along each survey's long axis with a buffer strip
      dropped at boundaries (see splits.assign_spatial_blocks).
    """
    if cfg.split.mode == "spatial_blocks":
        from .splits import assign_spatial_blocks

        records: list[TileRecord] = []
        for poly in cfg.split.polygons:
            records.extend(
                load_run_records(cfg.rot_dir(poly), poly, cfg.bands, cfg.base_dir, augmented=False)
            )
            if cfg.split.use_augmented_for_train:
                records.extend(
                    load_run_records(cfg.rotaug_dir(poly), poly, cfg.bands, cfg.base_dir, augmented=True)
                )
        splits = assign_spatial_blocks(
            records, tuple(cfg.split.fractions), cfg.split.buffer_m
        )
        _guard_no_augmented_in_eval(splits)
        return splits

    splits: dict[str, list[TileRecord]] = {}
    for split_name, polygons in (
        ("train", cfg.split.train),
        ("val", cfg.split.val),
        ("test", cfg.split.test),
    ):
        records: list[TileRecord] = []
        for poly in polygons:
            records.extend(
                load_run_records(cfg.rot_dir(poly), poly, cfg.bands, cfg.base_dir, augmented=False)
            )
            if split_name == "train" and cfg.split.use_augmented_for_train:
                records.extend(
                    load_run_records(cfg.rotaug_dir(poly), poly, cfg.bands, cfg.base_dir, augmented=True)
                )
        splits[split_name] = records

    _guard_no_augmented_in_eval(splits)
    return splits


def _guard_no_augmented_in_eval(splits: dict[str, list[TileRecord]]) -> None:
    """Hard guard: augmented tiles must be unreachable from val/test."""
    for split_name in ("val", "test"):
        if any(r.augmented for r in splits[split_name]):
            raise AssertionError(f"augmented tile leaked into {split_name} split")


def encode_target(
    label: np.ndarray,
    features: np.ndarray,
    class_ids: list[int],
    nodata: float,
    ignore_label: int,
) -> np.ndarray:
    """Label raster -> int64 target of model-channel indices, IGNORE_INDEX where untrainable.

    Ignored: background/unlabeled pixels (ignore_label) and any pixel where at
    least one feature band is nodata or NaN ("no data" must never be learned
    as "background seabed").
    """
    target = np.full(label.shape, IGNORE_INDEX, dtype=np.int64)
    for channel, class_id in enumerate(class_ids):
        target[label == class_id] = channel
    feature_invalid = np.any((features == nodata) | np.isnan(features), axis=0)
    target[feature_invalid] = IGNORE_INDEX
    return target


def features_by_polygon(
    splits: dict[str, list[TileRecord]]
) -> dict[str, list[np.ndarray]]:
    """Base-tile feature arrays grouped by polygon (input to normalize.compute_stats).

    Only non-augmented tiles contribute, so stats are independent of how many
    augmentation passes were generated.
    """
    grouped: dict[str, list[np.ndarray]] = {}
    for records in splits.values():
        for r in records:
            if not r.augmented:
                grouped.setdefault(r.polygon, []).append(r.features)
    return grouped
