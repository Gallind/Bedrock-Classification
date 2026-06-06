# tiling/tests/test_rotation.py
"""Tests for the rotation-aware geometry utilities."""
import math
import tempfile
from pathlib import Path

import pytest
from affine import Affine
import geopandas as gpd
from shapely.geometry import Polygon, box

from seabed_tiler.rotation import RotatedTileWindow, build_tile_affine, compute_label_footprint, minimum_bounding_rect


def test_rotated_tile_window_fields():
    w = RotatedTileWindow(row=0, col=1, u_origin=100.0, v_origin=200.0, theta=math.pi / 4)
    assert w.row == 0
    assert w.col == 1
    assert w.u_origin == pytest.approx(100.0)
    assert w.v_origin == pytest.approx(200.0)
    assert w.theta == pytest.approx(math.pi / 4)


def test_axis_aligned_affine_theta_zero():
    """theta=0 must produce a standard North-up rasterio affine."""
    # Tile origin at UTM (500, 1000) in the rotated frame
    # At theta=0 the local frame aligns with UTM so ox=500, oy=1000
    w = RotatedTileWindow(row=0, col=0, u_origin=500.0, v_origin=1000.0, theta=0.0)
    res = 1.0
    aff = build_tile_affine(w, res)
    expected = Affine(res, 0.0, 500.0, 0.0, -res, 1000.0)
    assert aff.a == pytest.approx(expected.a, abs=1e-9)
    assert aff.b == pytest.approx(expected.b, abs=1e-9)
    assert aff.c == pytest.approx(expected.c, abs=1e-9)
    assert aff.d == pytest.approx(expected.d, abs=1e-9)
    assert aff.e == pytest.approx(expected.e, abs=1e-9)
    assert aff.f == pytest.approx(expected.f, abs=1e-9)


def test_rotated_affine_90deg():
    """At theta=90 deg: moving col+1 steps North, moving row+1 steps East."""
    theta = math.pi / 2
    # u_origin=0, v_origin=0 -> ox=0*cos-0*sin=0, oy=0*sin+0*cos=0
    w = RotatedTileWindow(row=0, col=0, u_origin=0.0, v_origin=0.0, theta=theta)
    res = 2.0
    aff = build_tile_affine(w, res)
    # a = res*cos(90)=0, b = res*sin(90)=res, d = res*sin(90)=res, e = -res*cos(90)=0
    assert aff.a == pytest.approx(0.0, abs=1e-9)
    assert aff.b == pytest.approx(res, abs=1e-9)
    assert aff.d == pytest.approx(res, abs=1e-9)
    assert aff.e == pytest.approx(0.0, abs=1e-9)


def _write_shp(geoms, path: Path):
    """Write a shapefile with the given geometries."""
    gdf = gpd.GeoDataFrame({"geometry": geoms}, crs="EPSG:32636")
    gdf.to_file(path, driver="ESRI Shapefile")


def test_label_footprint_single_polygon():
    with tempfile.TemporaryDirectory() as tmp:
        shp = Path(tmp) / "labels.shp"
        poly = box(100.0, 200.0, 300.0, 500.0)
        _write_shp([poly], shp)
        fp = compute_label_footprint([shp])
        assert isinstance(fp, Polygon)
        assert fp.contains(poly) or fp.equals(poly)


def test_label_footprint_multiple_shapefiles():
    with tempfile.TemporaryDirectory() as tmp:
        shp1 = Path(tmp) / "rock.shp"
        shp2 = Path(tmp) / "sand.shp"
        _write_shp([box(0, 0, 10, 10)], shp1)
        _write_shp([box(20, 0, 30, 10)], shp2)
        fp = compute_label_footprint([shp1, shp2])
        assert fp.bounds[2] > 20  # footprint spans both shapes


def test_label_footprint_raises_on_empty():
    with tempfile.TemporaryDirectory() as tmp:
        shp = Path(tmp) / "empty.shp"
        gdf = gpd.GeoDataFrame({"geometry": []}, crs="EPSG:32636")
        gdf.to_file(shp, driver="ESRI Shapefile")
        with pytest.raises(ValueError, match="no valid annotation geometries"):
            compute_label_footprint([shp])


def test_mbr_long_edge_used_for_theta():
    """For a rectangle that is twice as long as it is wide, theta must align with the long side."""
    angle = math.radians(30)
    c, s = math.cos(angle), math.sin(angle)
    L, W = 200.0, 50.0
    corners = [
        (0.0, 0.0),
        (L * c, L * s),
        (L * c - W * s, L * s + W * c),
        (-W * s, W * c),
    ]
    poly = Polygon(corners)
    _, theta, mbr_corners = minimum_bounding_rect(poly)
    # theta should be near 30 degrees (the long axis)
    assert abs(math.degrees(theta) - 30.0) < 2.0
    assert mbr_corners.shape == (4, 2)


def test_mbr_small_theta_falls_back_to_zero():
    """A 3-degree rotation triggers the axis-aligned fallback."""
    angle = math.radians(3)
    c, s = math.cos(angle), math.sin(angle)
    L, W = 300.0, 80.0
    corners = [(0, 0), (L * c, L * s), (L * c - W * s, L * s + W * c), (-W * s, W * c)]
    poly = Polygon(corners)
    _, theta, _ = minimum_bounding_rect(poly)
    assert theta == pytest.approx(0.0)


def test_mbr_near_90deg_falls_back_to_zero():
    """An 87-degree rotation triggers the axis-aligned fallback."""
    angle = math.radians(87)
    c, s = math.cos(angle), math.sin(angle)
    L, W = 300.0, 80.0
    corners = [(0, 0), (L * c, L * s), (L * c - W * s, L * s + W * c), (-W * s, W * c)]
    poly = Polygon(corners)
    _, theta, _ = minimum_bounding_rect(poly)
    assert theta == pytest.approx(0.0)


def test_mbr_returns_polygon_and_corners():
    """Return types: (Polygon, float, ndarray shape (4,2))."""
    rect = box(0, 0, 50, 30)
    mbr, theta, corners = minimum_bounding_rect(rect)
    assert isinstance(mbr, Polygon)
    assert isinstance(theta, float)
    assert corners.shape == (4, 2)
