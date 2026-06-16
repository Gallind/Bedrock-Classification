"""losses.py: masking semantics, gradient isolation, class weights."""

import numpy as np
import pytest
import torch

from seabed_unet.data import IGNORE_INDEX
from seabed_unet.losses import (
    MaskedSegmentationLoss,
    compute_class_weights,
    soft_dice_loss,
)


def test_perfect_prediction_gives_near_zero_loss():
    target = torch.randint(0, 3, (2, 8, 8))
    logits = torch.full((2, 3, 8, 8), -100.0)
    logits.scatter_(1, target.unsqueeze(1), 100.0)  # huge margin on the true class
    loss = MaskedSegmentationLoss(0.5, 0.5)(logits, target)
    assert loss.item() < 0.01


def test_ignored_pixels_receive_zero_gradient():
    target = torch.randint(0, 3, (1, 8, 8))
    target[0, :4, :] = IGNORE_INDEX  # mask the top half
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    loss = MaskedSegmentationLoss(0.5, 0.5)(logits, target)
    loss.backward()
    assert torch.all(logits.grad[0, :, :4, :] == 0.0), "masked pixels leaked gradient"
    assert torch.any(logits.grad[0, :, 4:, :] != 0.0)


def test_fully_masked_batch_is_finite_zero():
    target = torch.full((1, 8, 8), IGNORE_INDEX)
    logits = torch.randn(1, 3, 8, 8, requires_grad=True)
    loss = MaskedSegmentationLoss(0.5, 0.5)(logits, target)
    assert loss.item() == 0.0
    loss.backward()  # must not raise / produce NaN
    assert torch.isfinite(logits.grad).all()


def test_dice_loss_in_unit_range():
    logits = torch.randn(2, 3, 8, 8)
    target = torch.randint(0, 3, (2, 8, 8))
    loss = soft_dice_loss(logits, target)
    assert 0.0 <= loss.item() <= 1.0


def test_class_weights_change_the_loss():
    # Mixed-class target: re-weighting shifts the per-class balance of the mean.
    # (With a single-class target the weight cancels in CE's weighted mean.)
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 8, 8)
    target = torch.zeros(1, 8, 8, dtype=torch.int64)
    target[0, 4:, :] = 1
    flat = MaskedSegmentationLoss(1.0, 0.0)(logits, target)
    weighted = MaskedSegmentationLoss(
        1.0, 0.0, class_weights=torch.tensor([10.0, 1.0, 1.0])
    )(logits, target)
    assert flat.item() != pytest.approx(weighted.item())


def test_compute_class_weights_inverse_frequency():
    # 100 px of class 0, 50 of class 1, class 2 absent, some ignored
    target = np.concatenate(
        [np.zeros(100), np.ones(50), np.full(7, IGNORE_INDEX)]
    ).astype(np.int64)
    weights = compute_class_weights([target], num_classes=3)
    assert weights[2] == 0.0                       # absent class
    assert weights[1] == pytest.approx(2 * weights[0])  # half as frequent -> double weight
    # "balanced" invariant: every present class contributes equally overall
    assert weights[0] * 100 == pytest.approx(weights[1] * 50)
