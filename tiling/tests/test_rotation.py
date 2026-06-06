# tiling/tests/test_rotation.py
"""Tests for the rotation-aware geometry utilities."""
import math

import numpy as np
import pytest
from affine import Affine
from seabed_tiler.rotation import RotatedTileWindow, build_tile_affine


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
