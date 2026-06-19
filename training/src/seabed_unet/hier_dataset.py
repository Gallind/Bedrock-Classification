"""Label-remapping datasets for the two-stage hierarchical U-Net.

Stage 1: rock (ch 0) -> 0, non-rock (ch 1, 2) -> 1. All tiles used.
Stage 2: shallow_rock (ch 1) -> 0, sand (ch 2) -> 1. Rock pixels masked (IGNORE_INDEX).
         Only tiles that contain at least one non-rock labeled pixel are included.
"""

from __future__ import annotations

import numpy as np
import torch

from seabed_tiler.augment import augment_pair, random_d4

from .data import IGNORE_INDEX, TileRecord, encode_target
from .normalize import apply_stats

# Channel indices produced by encode_target with class_ids=[1,2,3]
ROCK_CH = 0
SHALLOW_CH = 1
SAND_CH = 2


def remap_stage1(target: np.ndarray) -> np.ndarray:
    """rock->0, non-rock->1, IGNORE stays IGNORE."""
    out = np.where(target == IGNORE_INDEX, IGNORE_INDEX,
           np.where(target == ROCK_CH, 0, 1)).astype(np.int64)
    return out


def remap_stage2(target: np.ndarray) -> np.ndarray:
    """shallow_rock->0, sand->1, rock pixels->IGNORE_INDEX."""
    out = np.where(target == IGNORE_INDEX, IGNORE_INDEX,
           np.where(target == ROCK_CH, IGNORE_INDEX,
           np.where(target == SHALLOW_CH, 0, 1))).astype(np.int64)
    return out


class Stage1Dataset(torch.utils.data.Dataset):
    """Binary rock/non-rock dataset — all tiles."""

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
            feat = apply_stats(r.features, r.polygon, bands, stats, nodata, band_modes)
            tgt = encode_target(r.label, r.features, class_ids, nodata, ignore_label)
            self.inputs.append(feat)
            self.targets.append(remap_stage1(tgt))
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


class Stage2Dataset(torch.utils.data.Dataset):
    """Binary shallow_rock/sand dataset — only tiles with non-rock pixels."""

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
        skipped = 0
        for r in records:
            feat = apply_stats(r.features, r.polygon, bands, stats, nodata, band_modes)
            tgt = encode_target(r.label, r.features, class_ids, nodata, ignore_label)
            tgt2 = remap_stage2(tgt)
            # Only include tiles that have at least one non-ignored pixel for stage 2
            if (tgt2 != IGNORE_INDEX).any():
                self.inputs.append(feat)
                self.targets.append(tgt2)
            else:
                skipped += 1
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self._skipped = skipped

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
