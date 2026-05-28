import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seabed_tiler.config import LabelRule
from seabed_tiler.labels import classify, normalize_name

CLASSES = {"rock": 1, "shallow_rock": 2, "sand": 3}
RULES = [
    LabelRule(pattern="shallow", **{"class": "shallow_rock"}),
    LabelRule(pattern="rock", **{"class": "rock"}),
    LabelRule(pattern="sand", **{"class": "sand"}),
]


def test_normalize_collapses_whitespace_and_case():
    assert normalize_name("  Class 2-  shallow rock ") == "class 2- shallow rock"


def test_every_real_name_variant_maps_correctly():
    # Exact strings observed in polygon1/3 classes.dbf.
    cases = {
        "Class 1- rock": 1,
        "Class 2- shallow rock": 2,
        "Class 2-  shallow rock": 2,   # double space
        "class2 -shallow rock": 2,     # no space after 'class2'
        "class 3 - sand": 3,
    }
    for name, expected in cases.items():
        assert classify(name, RULES, CLASSES) == expected, name


def test_shallow_takes_priority_over_rock():
    # "shallow rock" must not be caught by the bare "rock" rule.
    assert classify("shallow rock", RULES, CLASSES) == 2


def test_unmatched_name_returns_none():
    assert classify("seagrass meadow", RULES, CLASSES) is None
