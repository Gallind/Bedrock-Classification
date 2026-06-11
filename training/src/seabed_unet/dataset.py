"""Torch Dataset over normalized, target-encoded tile records.

D4 augmentation is delegated to seabed_tiler.augment (the canonical, tested
implementation — PYTHONPATH must include tiling/src). Per the contract, D4 is
training-time only; construct val/test datasets with augment=False.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from seabed_tiler.augment import augment_pair, random_d4

from .data import TileRecord, encode_target
from .normalize import apply_stats


class TileDataset(Dataset):
    """In-RAM dataset: all tiles preprocessed once at construction (~300 MB total)."""

    def __init__(
        self,
        records: list[TileRecord],
        bands: list[str],
        class_ids: list[int],
        stats: dict,
        nodata: float,
        ignore_label: int,
        band_modes: dict[str, str],
        augment: bool = False,
        seed: int = 0,
    ):
        self.inputs: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        for r in records:
            self.inputs.append(
                apply_stats(r.features, r.polygon, bands, stats, nodata, band_modes)
            )
            self.targets.append(
                encode_target(r.label, r.features, class_ids, nodata, ignore_label)
            )
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.inputs[idx], self.targets[idx]
        if self.augment:
            x, y = augment_pair(x, y, random_d4(self.rng))
        return (
            torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(y)),
        )
