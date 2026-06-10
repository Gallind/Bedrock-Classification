# tiling/tests/test_rotation.py
"""Tests for the rotation-aware geometry utilities."""
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest
from affine import Affine
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from rasterio.crs import CRS

from seabed_tiler.rotation import RotatedTileWindow, build_tile_affine, compute_label_footprint, minimum_bounding_rect, build_rotated_windows, extract_rotated_tile
from seabed_tiler.manifest import write_rotated_manifest


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


def test_build_rotated_windows_stride():
    """Adjacent column neighbours differ in u_origin by exactly stride_m."""
    corners = np.array([(0.0, 0.0), (256.0, 0.0), (256.0, 256.0), (0.0, 256.0)])
    wins = build_rotated_windows(corners, theta=0.0, tile_size_m=128.0, stride_m=64.0)
    row0 = [w for w in wins if w.row == 0]
    assert len(row0) >= 2
    assert row0[1].u_origin - row0[0].u_origin == pytest.approx(64.0)


def test_build_rotated_windows_all_inside_mbr():
    """All tile corners must stay within the MBR bounds (full containment, no partial edges)."""
    corners = np.array([(100.0, 200.0), (500.0, 200.0), (500.0, 600.0), (100.0, 600.0)])
    tile_size = 100.0
    stride = 50.0
    wins = build_rotated_windows(corners, theta=0.0, tile_size_m=tile_size, stride_m=stride)
    for w in wins:
        assert w.u_origin >= 100.0 - 1e-6
        assert w.u_origin + tile_size <= 500.0 + 1e-6
        assert w.v_origin - tile_size >= 200.0 - 1e-6
        assert w.v_origin <= 600.0 + 1e-6


def test_build_rotated_windows_stores_theta():
    """All windows must carry the theta used for the grid."""
    corners = np.array([(0.0, 0.0), (256.0, 0.0), (256.0, 256.0), (0.0, 256.0)])
    wins = build_rotated_windows(corners, theta=0.3, tile_size_m=128.0, stride_m=64.0)
    assert len(wins) > 0
    assert all(w.theta == pytest.approx(0.3) for w in wins)
    assert all(isinstance(w, RotatedTileWindow) for w in wins)


def test_build_rotated_windows_zero_offset_reproduces_default_grid():
    """u_offset=v_offset=0 must produce exactly the same windows as no offsets
    (regression guard for the augmentation origin-shift feature)."""
    corners = np.array([(0.0, 0.0), (256.0, 0.0), (256.0, 256.0), (0.0, 256.0)])
    base = build_rotated_windows(corners, theta=0.2, tile_size_m=128.0, stride_m=64.0)
    shifted = build_rotated_windows(
        corners, theta=0.2, tile_size_m=128.0, stride_m=64.0,
        u_offset=0.0, v_offset=0.0,
    )
    assert base == shifted


def test_build_rotated_windows_offset_shifts_all_origins():
    """A (u, v) origin offset must shift every window by exactly that amount."""
    corners = np.array([(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)])
    base = build_rotated_windows(corners, theta=0.0, tile_size_m=100.0, stride_m=50.0)
    shifted = build_rotated_windows(
        corners, theta=0.0, tile_size_m=100.0, stride_m=50.0,
        u_offset=25.0, v_offset=-10.0,
    )
    assert len(shifted) > 0
    base_by_rc = {(w.row, w.col): w for w in base}
    for w in shifted:
        b = base_by_rc.get((w.row, w.col))
        if b is None:
            continue  # shifted grid may drop trailing tiles near the far edge
        assert w.u_origin - b.u_origin == pytest.approx(25.0)
        assert w.v_origin - b.v_origin == pytest.approx(-10.0)


def test_build_rotated_windows_offset_keeps_full_containment():
    """Offset windows must still lie fully inside the MBR (no partial tiles)."""
    corners = np.array([(100.0, 200.0), (500.0, 200.0), (500.0, 600.0), (100.0, 600.0)])
    tile_size = 100.0
    wins = build_rotated_windows(
        corners, theta=0.0, tile_size_m=tile_size, stride_m=50.0,
        u_offset=30.0, v_offset=-20.0,
    )
    assert len(wins) > 0
    for w in wins:
        assert w.u_origin >= 100.0 - 1e-6
        assert w.u_origin + tile_size <= 500.0 + 1e-6
        assert w.v_origin - tile_size >= 200.0 - 1e-6
        assert w.v_origin <= 600.0 + 1e-6


def _tile_utm_corners(w: RotatedTileWindow, tile_size: float):
    """UTM coords of a window's 4 corners (u, v rotated frame -> x, y)."""
    c, s = math.cos(w.theta), math.sin(w.theta)
    for du in (0.0, tile_size):
        for dv in (0.0, -tile_size):
            u, v = w.u_origin + du, w.v_origin + dv
            yield u * c - v * s, u * s + v * c


def test_build_rotated_windows_jittered_theta_stays_inside_mbr():
    """When the grid angle differs from the MBR's own orientation (augmentation
    theta jitter), tiles must lie inside the MBR polygon itself -- not inside the
    larger axis-aligned bounding box of the MBR in the jittered frame. Tiles
    outside the MBR fall outside the surveyed data and produce dead pixels."""
    corners = np.array([(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)])
    mbr = Polygon(corners).buffer(1e-6)
    tile_size = 100.0
    theta = math.radians(8.0)  # MBR is axis-aligned; the grid is jittered by 8 deg
    wins = build_rotated_windows(corners, theta=theta, tile_size_m=tile_size, stride_m=50.0)
    assert len(wins) > 0
    for w in wins:
        for x, y in _tile_utm_corners(w, tile_size):
            assert mbr.contains(Point(x, y)), (
                f"tile r{w.row:03d} c{w.col:03d} corner ({x:.1f}, {y:.1f}) outside MBR"
            )


def test_build_rotated_windows_jittered_theta_with_offset_stays_inside_mbr():
    """Theta jitter combined with origin shifts (augmentation pass 4 shape) must
    also keep every tile inside the MBR polygon."""
    corners = np.array([(0.0, 0.0), (400.0, 0.0), (400.0, 400.0), (0.0, 400.0)])
    mbr = Polygon(corners).buffer(1e-6)
    tile_size = 100.0
    theta = math.radians(-4.0)
    wins = build_rotated_windows(
        corners, theta=theta, tile_size_m=tile_size, stride_m=50.0,
        u_offset=12.5, v_offset=-12.5,
    )
    assert len(wins) > 0
    for w in wins:
        for x, y in _tile_utm_corners(w, tile_size):
            assert mbr.contains(Point(x, y)), (
                f"tile r{w.row:03d} c{w.col:03d} corner ({x:.1f}, {y:.1f}) outside MBR"
            )


def test_build_rotated_windows_narrow_mbr_small_jitter_finds_tiles():
    """A narrow elongated MBR (polygon5 shape: 872 x 157 m, 128 m tiles) must still
    yield close to a full row of tiles under a small theta jitter. A corner-anchored
    grid misses the strip entirely (0 windows) because its quantized row positions
    fall outside the tilted strip; the anchor phase must be chosen so tiles land
    inside."""
    angle = math.radians(-33.0)
    c, s = math.cos(angle), math.sin(angle)
    L, W = 872.0, 157.0
    corners = np.array([
        (0.0, 0.0),
        (L * c, L * s),
        (L * c - W * s, L * s + W * c),
        (-W * s, W * c),
    ])
    mbr = Polygon(corners).buffer(1e-6)
    tile_size = 128.0
    for jitter_deg in (1.0, -1.0, 2.0, -2.0):
        theta = angle + math.radians(jitter_deg)
        wins = build_rotated_windows(corners, theta=theta, tile_size_m=tile_size, stride_m=64.0)
        assert len(wins) >= 8, f"jitter {jitter_deg} deg: only {len(wins)} windows"
        for w in wins:
            for x, y in _tile_utm_corners(w, tile_size):
                assert mbr.contains(Point(x, y))


def test_build_rotated_windows_aligned_grid_keeps_corner_anchor():
    """When the grid angle matches the MBR orientation, the anchor must stay at the
    MBR corner (u_min, v_max) so existing base grids are reproduced exactly."""
    corners = np.array([(100.0, 200.0), (500.0, 200.0), (500.0, 600.0), (100.0, 600.0)])
    wins = build_rotated_windows(corners, theta=0.0, tile_size_m=128.0, stride_m=64.0)
    assert len(wins) > 0
    assert min(w.u_origin for w in wins) == pytest.approx(100.0)
    assert max(w.v_origin for w in wins) == pytest.approx(600.0)


def test_extract_rotated_tile_theta_zero_equals_direct_slice():
    """At theta=0 the extracted tile must equal a direct numpy array slice."""
    from rasterio.crs import CRS
    from rasterio.enums import Resampling

    res = 1.0
    crs = CRS.from_epsg(32636)
    src = np.arange(400.0, dtype="float32").reshape(1, 20, 20)  # 1 band, 20x20
    # transform: col 0 -> x=0, row 0 -> y=20  (North-up, 1m pixels)
    src_transform = Affine(res, 0.0, 0.0, 0.0, -res, 20.0)
    nodata = -9999.0

    # Tile starting at local (u=5, v=15), theta=0, 10 pixels wide
    # At theta=0: u=5 -> col=5 (x=u), v=15 -> row=20-15=5
    win = RotatedTileWindow(row=0, col=0, u_origin=5.0, v_origin=15.0, theta=0.0)
    tile, tile_transform = extract_rotated_tile(
        src, src_transform, crs, win, res, nodata,
        tile_px=10, resampling=Resampling.nearest,
    )
    expected = src[0, 5:15, 5:15]
    np.testing.assert_array_equal(tile[0], expected)
    # tile_transform.c is the x-origin (= u_origin at theta=0)
    assert tile_transform.c == pytest.approx(5.0)
    # tile_transform.f is the y-origin (= v_origin at theta=0)
    assert tile_transform.f == pytest.approx(15.0)


def test_write_rotated_manifest_creates_csv_and_geojson():
    """write_rotated_manifest must produce a CSV and a GeoJSON with 4-corner polygons."""
    crs = CRS.from_epsg(32636)
    res = 1.0
    tile_px = 10
    # theta=0: tile at u_origin=500, v_origin=1000 -> 10x10px -> corners at (500,990),(510,990),(510,1000),(500,1000)
    rows = [{
        "tile_id": "polygon1_r000_c000",
        "row": 0, "col": 0,
        "theta_deg": 0.0,
        "u_origin": 500.0, "v_origin": 1000.0,
        "valid_frac": 0.9,
        "features_path": "tiles/features/polygon1_r000_c000.tif",
        "label_path": "tiles/labels/polygon1_r000_c000.tif",
        "background_px": 5, "rock_px": 85, "shallow_rock_px": 10, "sand_px": 0,
    }]
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp)
        write_rotated_manifest(rows, out_dir, crs, res, tile_px)
        assert (out_dir / "manifest.csv").exists()
        geojson_path = out_dir / "manifest.geojson"
        assert geojson_path.exists()
        with open(geojson_path) as f:
            gj = json.load(f)
        assert gj["type"] == "FeatureCollection"
        assert len(gj["features"]) == 1
        geom = gj["features"][0]["geometry"]
        assert geom["type"] == "Polygon"
        assert len(geom["coordinates"][0]) == 5  # 4 corners + closing repeat
