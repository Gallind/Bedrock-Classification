"""Masked segmentation loss: weighted cross-entropy + soft Dice.

Every term respects IGNORE_INDEX (-1): background/unlabeled pixels and pixels
with invalid features contribute neither loss nor gradient — the contract's
defence against teaching the model that "no data" means "background seabed".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import IGNORE_INDEX


def compute_class_weights(targets: list[np.ndarray], num_classes: int) -> np.ndarray:
    """Inverse-frequency ("balanced") class weights from TRAIN targets only.

    w_i = total / (n_present * count_i), so every present class contributes the
    same total weight. Classes absent from the training pixels get weight 0
    (nothing to learn from).
    """
    counts = np.zeros(num_classes, dtype=np.int64)
    for target in targets:
        valid = target[target != IGNORE_INDEX]
        counts += np.bincount(valid, minlength=num_classes)
    weights = np.zeros(num_classes, dtype=np.float32)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    return weights


def soft_dice_loss(
    logits: torch.Tensor, target: torch.Tensor, eps: float = 1.0
) -> torch.Tensor:
    """1 - mean per-class soft Dice over non-ignored pixels.

    logits: (N, C, H, W); target: (N, H, W) with IGNORE_INDEX for masked pixels.
    """
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1)
    valid = (target != IGNORE_INDEX).unsqueeze(1).float()  # (N,1,H,W)
    safe_target = target.clamp(min=0)
    onehot = (
        F.one_hot(safe_target, num_classes).permute(0, 3, 1, 2).float() * valid
    )
    probs = probs * valid

    dims = (0, 2, 3)
    intersection = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


class MaskedSegmentationLoss(nn.Module):
    """ce_weight * weighted-CE(ignore_index) + dice_weight * soft Dice."""

    def __init__(
        self,
        ce_weight: float,
        dice_weight: float,
        class_weights: torch.Tensor | None = None,
    ):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not (target != IGNORE_INDEX).any():
            # Fully-masked batch (pathological): zero loss, keeps graph intact.
            return logits.sum() * 0.0
        loss = logits.new_zeros(())
        if self.ce_weight > 0:
            loss = loss + self.ce_weight * self.ce(logits, target)
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * soft_dice_loss(logits, target)
        return loss
