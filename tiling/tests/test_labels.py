import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
from shapely.geometry import LineString, Polygon

from seabed_tiler.config import LabelRule
from seabed_tiler.labels import (
    _feature_to_polygons,
    _group_shapefile_features,
    _ordered_shapes,
    classify,
    normalize_name,
)

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


# Per-class label path (polygons 3/4/5): _feature_to_polygons.
SQUARE = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]  # closed ring
OPEN = [(0, 0), (0, 10), (10, 10), (10, 0)]            # open path


def test_polygonize_closes_a_ring_linestring():
    polys = _feature_to_polygons(LineString(SQUARE), polygonize=True)
    assert len(polys) == 1
    assert polys[0].area == 100.0


def test_open_linestring_is_dropped():
    assert _feature_to_polygons(LineString(OPEN), polygonize=True) == []


def test_linestring_ignored_when_polygonize_disabled():
    assert _feature_to_polygons(LineString(SQUARE), polygonize=False) == []


def test_polygon_passes_through():
    out = _feature_to_polygons(Polygon(SQUARE), polygonize=False)
    assert len(out) == 1
    assert out[0].area == 100.0


def test_bowtie_polygon_is_repaired_to_valid_area():
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10), (0, 0)])
    out = _feature_to_polygons(bowtie, polygonize=False)
    assert out
    assert all(p.is_valid and p.area > 0 for p in out)


# Snap tolerance: rings the annotator left open by a small gap (polygon3's two
# biggest annotations are open by 0.1-1.1 m) must close when within tolerance.

NEARLY_CLOSED = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 1)]  # 1 m short of closing


def test_nearly_closed_ring_snaps_within_tolerance():
    polys = _feature_to_polygons(
        LineString(NEARLY_CLOSED), polygonize=True, close_tolerance_m=2.0
    )
    assert len(polys) == 1
    assert polys[0].area == pytest.approx(100.0, rel=0.1)


def test_nearly_closed_ring_dropped_at_default_zero_tolerance():
    assert _feature_to_polygons(LineString(NEARLY_CLOSED), polygonize=True) == []


def test_gap_beyond_tolerance_still_dropped():
    wide_open = [(0, 0), (10, 0), (10, 10), (0, 10), (0, 5)]  # 5 m gap
    assert _feature_to_polygons(
        LineString(wide_open), polygonize=True, close_tolerance_m=2.0
    ) == []


def test_exactly_closed_ring_unaffected_by_tolerance():
    polys = _feature_to_polygons(
        LineString(SQUARE), polygonize=True, close_tolerance_m=2.0
    )
    assert len(polys) == 1
    assert polys[0].area == pytest.approx(100.0)


# Shapefile path (polygon1): classify + group, then burn in priority order.
# Recovers the feature whose class name was overwritten by its area string.
RULES_WITH_AREA_FALLBACK = RULES + [LabelRule(pattern=r"0\.00501 sq km", **{"class": "rock"})]
PRIORITY = ["sand", "shallow_rock", "rock"]  # rock wins on overlap


def test_group_classifies_repairs_and_groups():
    feats = [
        ("Class 1- rock", Polygon(SQUARE)),
        ("class 3 - sand", Polygon(SQUARE)),
        ("Class 2- shallow rock", Polygon(SQUARE)),
    ]
    by_class, unmatched = _group_shapefile_features(feats, RULES, CLASSES)
    assert unmatched == []
    assert set(by_class) == {1, 2, 3}
    assert all(len(v) == 1 for v in by_class.values())


def test_group_reports_unmatched_name_instead_of_dropping_silently():
    feats = [("seagrass meadow", Polygon(SQUARE)), ("Class 1- rock", Polygon(SQUARE))]
    by_class, unmatched = _group_shapefile_features(feats, RULES, CLASSES)
    assert unmatched == ["seagrass meadow"]
    assert set(by_class) == {1}  # only the rock feature grouped


def test_group_area_string_rule_recovers_rock():
    # NAME overwritten by area string; the fallback rule maps it to rock.
    feats = [("0.00501 sq km", Polygon(SQUARE))]
    by_class, unmatched = _group_shapefile_features(feats, RULES_WITH_AREA_FALLBACK, CLASSES)
    assert unmatched == []
    assert set(by_class) == {1}


def test_priority_burn_order_lets_rock_win_on_overlap():
    # rock + shallow over the same area; rock must be burned LAST so it overwrites.
    by_class = {2: [Polygon(SQUARE)], 1: [Polygon(SQUARE)]}
    shapes = _ordered_shapes(by_class, CLASSES, PRIORITY)
    assert shapes[-1][1] == CLASSES["rock"]  # rock burned last == wins on overlap
