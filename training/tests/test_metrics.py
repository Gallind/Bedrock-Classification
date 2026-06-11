"""metrics.py: confusion matrix masking + hand-computed reference values."""

import numpy as np
import pytest

from seabed_unet.data import IGNORE_INDEX
from seabed_unet.metrics import (
    cohens_kappa,
    confusion_matrix,
    dice_per_class,
    metrics_report,
    overall_accuracy,
    producers_accuracy,
    users_accuracy,
)

# Reference confusion matrix (rows = true, cols = pred), hand-computed values:
# total=10, po=8/10; rows=[3,3,4], cols=[3,4,3]; pe=(9+12+12)/100=0.33
CM = np.array([[2, 1, 0], [0, 3, 0], [1, 0, 3]], dtype=np.int64)


def test_confusion_matrix_ignores_masked_pixels():
    target = np.array([0, 1, 2, IGNORE_INDEX, IGNORE_INDEX])
    pred = np.array([0, 2, 2, 0, 1])  # predictions at ignored pixels are irrelevant
    cm = confusion_matrix(pred, target, 3)
    assert cm.sum() == 3
    assert cm[0, 0] == 1 and cm[1, 2] == 1 and cm[2, 2] == 1


def test_overall_accuracy():
    assert overall_accuracy(CM) == pytest.approx(0.8)


def test_cohens_kappa_hand_computed():
    assert cohens_kappa(CM) == pytest.approx((0.8 - 0.33) / (1 - 0.33))


def test_dice_per_class_hand_computed():
    dice = dice_per_class(CM)
    assert dice[0] == pytest.approx(2 * 2 / (3 + 3))
    assert dice[1] == pytest.approx(2 * 3 / (3 + 4))
    assert dice[2] == pytest.approx(2 * 3 / (4 + 3))


def test_producers_and_users_accuracy_hand_computed():
    assert producers_accuracy(CM) == pytest.approx([2 / 3, 1.0, 3 / 4])
    assert users_accuracy(CM) == pytest.approx([2 / 3, 3 / 4, 1.0])


def test_absent_class_yields_nan_not_crash():
    cm = np.array([[5, 0], [0, 0]], dtype=np.int64)  # class 1 never occurs
    assert np.isnan(dice_per_class(cm)[1])
    assert np.isnan(producers_accuracy(cm)[1])


def test_metrics_report_structure():
    report = metrics_report(CM, ["rock", "shallow_rock", "sand"])
    assert report["overall_accuracy"] == pytest.approx(0.8)
    assert set(report["per_class"]) == {"rock", "shallow_rock", "sand"}
    assert report["per_class"]["rock"]["support_px"] == 3
    assert report["confusion_matrix"] == CM.tolist()
    # JSON-serializable end to end
    import json

    json.dumps(report)
