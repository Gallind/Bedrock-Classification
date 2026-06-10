# tiling/tests/test_augmented_tiling.py
"""Integration tests for the source-level augmentation tiling loop."""
import math
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.affinity import rotate
from shapely.geometry import Point, box

from seabed_tiler.config import Config
from seabed_tiler.rotated_tiler import run_augmented_tiling, run_rotated_tiling


GRID_PX = 64  # synthetic master grid is 64x64 at 1 m


@pytest.fixture
def cfg_and_grid(tmp_path):
    """A minimal synthetic config + master grid pair.

    Master grid: 64x64 px at 1 m, UTM-like coords x in [0, 64], y in [0, 64].
    Labels: a single square polygon covering most of the grid (theta -> 0 fallback).
    """
    src_dir = tmp_path / "DataBase" / "testpoly"
    src_dir.mkdir(parents=True)
    gdf = gpd.GeoDataFrame(
        {"NAME": ["rock"], "geometry": [box(4.0, 4.0, 60.0, 60.0)]},
        crs="EPSG:32636",
    )
    gdf.to_file(src_dir / "labels.shp", driver="ESRI Shapefile")

    cfg = Config(
        name="testpoly",
        src_dir="DataBase/testpoly",
        target_resolution_m=1.0,
        tile_size_m=16.0,
        overlap=0.5,
        band_order=["bathymetry"],
        layers=[{"name": "bathymetry", "kind": "xyz", "path": "bathy.xyz"}],
        labels={
            "kind": "shapefile",
            "path": "labels.shp",
            "classes": {"rock": 1, "shallow_rock": 2, "sand": 3},
            "rules": [{"pattern": "rock", "class": "rock"}],
        },
        augmentation={
            "enabled": True,
            "passes": [
                {"theta_offset_deg": 0.0, "u_shift_frac": 0.25, "v_shift_frac": 0.25},
                {"theta_offset_deg": 6.0, "u_shift_frac": 0.0, "v_shift_frac": 0.0},
            ],
        },
    )
    cfg.base_dir = tmp_path

    rng = np.random.default_rng(0)
    features = rng.normal(-30.0, 2.0, size=(GRID_PX, GRID_PX)).astype("float32")
    label = np.ones((GRID_PX, GRID_PX), dtype="uint8")
    grid = {
        "transform": Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(GRID_PX)),
        "crs": CRS.from_epsg(32636),
        "nodata": -9999.0,
        "shape": (GRID_PX, GRID_PX),
        "features": {"bathymetry": features},
        "label": label,
    }
    return cfg, grid


def test_augmented_tiling_writes_to_rotaug_dir(cfg_and_grid):
    cfg, grid = cfg_and_grid
    rows, _ = run_augmented_tiling(cfg, grid)
    out_dir = cfg.base_dir / "outputs" / "testpoly" / (cfg.run_tag + "_rotaug")
    assert out_dir.exists()
    assert len(rows) > 0
    for r in rows:
        assert (cfg.base_dir / r["features_path"]).exists()
        assert (cfg.base_dir / r["label_path"]).exists()


def test_augmented_rows_carry_pass_provenance(cfg_and_grid):
    cfg, grid = cfg_and_grid
    rows, _ = run_augmented_tiling(cfg, grid)
    passes_seen = {r["aug_pass"] for r in rows}
    assert passes_seen == {1, 2}
    for r in rows:
        if r["aug_pass"] == 1:
            assert r["theta_offset_deg"] == pytest.approx(0.0)
            assert r["u_shift_frac"] == pytest.approx(0.25)
        else:
            assert r["theta_offset_deg"] == pytest.approx(6.0)
            assert r["u_shift_frac"] == pytest.approx(0.0)
        assert f"_p{r['aug_pass']:02d}_" in r["tile_id"]


def test_augmented_tile_ids_unique_across_passes(cfg_and_grid):
    cfg, grid = cfg_and_grid
    rows, _ = run_augmented_tiling(cfg, grid)
    ids = [r["tile_id"] for r in rows]
    assert len(ids) == len(set(ids))


def test_augmented_rows_have_tile_centers_inside_grid(cfg_and_grid):
    cfg, grid = cfg_and_grid
    rows, _ = run_augmented_tiling(cfg, grid)
    for r in rows:
        assert 0.0 <= r["center_x"] <= GRID_PX
        assert 0.0 <= r["center_y"] <= GRID_PX


def test_origin_shift_pass_moves_tiles_by_quarter_stride(cfg_and_grid):
    """Pass 1 (u/v shift 0.25 of an 8 m stride, theta offset 0) must place tile
    centers exactly 2 m away from the unshifted base grid tiles."""
    cfg, grid = cfg_and_grid
    base_rows, _ = run_rotated_tiling(cfg, grid)
    aug_rows, _ = run_augmented_tiling(cfg, grid)
    base_by_rc = {(r["row"], r["col"]): r for r in base_rows}
    pass1 = [r for r in aug_rows if r["aug_pass"] == 1]
    assert len(pass1) > 0
    checked = 0
    for r in pass1:
        b = base_by_rc.get((r["row"], r["col"]))
        if b is None:
            continue
        assert r["center_x"] - b["center_x"] == pytest.approx(2.0, abs=1e-6)
        assert r["center_y"] - b["center_y"] == pytest.approx(-2.0, abs=1e-6)
        checked += 1
    assert checked > 0


def test_base_rotated_manifest_also_has_centers(cfg_and_grid):
    cfg, grid = cfg_and_grid
    rows, _ = run_rotated_tiling(cfg, grid)
    assert len(rows) > 0
    for r in rows:
        assert "center_x" in r and "center_y" in r


def test_augmentation_disabled_raises(cfg_and_grid):
    cfg, grid = cfg_and_grid
    cfg.augmentation.enabled = False
    with pytest.raises(ValueError, match="augmentation"):
        run_augmented_tiling(cfg, grid)


def test_augmented_run_removes_stale_outputs_from_previous_runs(cfg_and_grid):
    """Re-running --augment must not leave tiles from a previous (different) run on
    disk: to_jpg converts every tif it finds, so orphans keep showing up in the JPEG
    previews even though the manifest no longer lists them."""
    cfg, grid = cfg_and_grid
    out = cfg.base_dir / "outputs" / "testpoly" / (cfg.run_tag + "_rotaug")
    stale = out / "tiles" / "features" / "testpoly_p99_r999_c999.tif"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    stale_jpg = out / "jpg" / "labels" / "testpoly_p99_r999_c999.jpg"
    stale_jpg.parent.mkdir(parents=True)
    stale_jpg.write_bytes(b"stale")
    rows, _ = run_augmented_tiling(cfg, grid)
    assert len(rows) > 0
    assert not stale.exists()
    assert not stale_jpg.exists()


def test_rotated_run_removes_stale_outputs_from_previous_runs(cfg_and_grid):
    cfg, grid = cfg_and_grid
    out = cfg.base_dir / "outputs" / "testpoly" / (cfg.run_tag + "_rot")
    stale = out / "tiles" / "labels" / "testpoly_r999_c999.tif"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    rows, _ = run_rotated_tiling(cfg, grid)
    assert len(rows) > 0
    assert not stale.exists()


def test_jittered_pass_tiles_stay_inside_label_mbr(cfg_and_grid):
    """Pass 2 (theta offset 6 deg) tiles must lie fully inside the annotation MBR.
    Tiles spilling past the MBR fall outside the surveyed data and ship dead
    (nodata) areas into the training set."""
    cfg, grid = cfg_and_grid
    rows, _ = run_augmented_tiling(cfg, grid)
    mbr = box(4.0, 4.0, 60.0, 60.0).buffer(1e-6)  # fixture label footprint
    pass2 = [r for r in rows if r["aug_pass"] == 2]
    assert len(pass2) > 0
    for r in pass2:
        c = math.cos(math.radians(r["theta_deg"]))
        s = math.sin(math.radians(r["theta_deg"]))
        for du in (0.0, cfg.tile_size_m):
            for dv in (0.0, -cfg.tile_size_m):
                u, v = r["u_origin"] + du, r["v_origin"] + dv
                x, y = u * c - v * s, u * s + v * c
                assert mbr.contains(Point(x, y)), (
                    f"{r['tile_id']} corner ({x:.1f}, {y:.1f}) outside annotation MBR"
                )


def test_base_and_augmented_runs_use_same_feature_resampling(tmp_path):
    """An augmentation pass with zero offsets reproduces the base rotated grid, so
    its feature tiles must be pixel-identical to the base run's. A mismatch means
    the two runs resample differently, making augmented tiles statistically
    distinguishable from base tiles (texture domain shift in the training set)."""
    src_dir = tmp_path / "DataBase" / "rotpoly"
    src_dir.mkdir(parents=True)
    footprint = rotate(box(12.0, 20.0, 52.0, 44.0), 30.0, origin="centroid")
    gdf = gpd.GeoDataFrame({"NAME": ["rock"], "geometry": [footprint]}, crs="EPSG:32636")
    gdf.to_file(src_dir / "labels.shp", driver="ESRI Shapefile")

    cfg = Config(
        name="rotpoly",
        src_dir="DataBase/rotpoly",
        target_resolution_m=1.0,
        tile_size_m=16.0,
        overlap=0.5,
        band_order=["bathymetry"],
        layers=[{"name": "bathymetry", "kind": "xyz", "path": "bathy.xyz"}],
        labels={
            "kind": "shapefile",
            "path": "labels.shp",
            "classes": {"rock": 1, "shallow_rock": 2, "sand": 3},
            "rules": [{"pattern": "rock", "class": "rock"}],
        },
        augmentation={
            "enabled": True,
            "passes": [{"theta_offset_deg": 0.0, "u_shift_frac": 0.0, "v_shift_frac": 0.0}],
        },
    )
    cfg.base_dir = tmp_path

    rng = np.random.default_rng(0)
    grid = {
        "transform": Affine(1.0, 0.0, 0.0, 0.0, -1.0, float(GRID_PX)),
        "crs": CRS.from_epsg(32636),
        "nodata": -9999.0,
        "shape": (GRID_PX, GRID_PX),
        "features": {"bathymetry": rng.normal(-30.0, 2.0, size=(GRID_PX, GRID_PX)).astype("float32")},
        "label": np.ones((GRID_PX, GRID_PX), dtype="uint8"),
    }

    base_rows, _ = run_rotated_tiling(cfg, grid)
    aug_rows, _ = run_augmented_tiling(cfg, grid)
    base_by_rc = {(r["row"], r["col"]): r for r in base_rows}
    checked = 0
    for r in aug_rows:
        b = base_by_rc.get((r["row"], r["col"]))
        if b is None:
            continue
        with rasterio.open(tmp_path / r["features_path"]) as src:
            aug_tile = src.read()
        with rasterio.open(tmp_path / b["features_path"]) as src:
            base_tile = src.read()
        np.testing.assert_array_equal(
            aug_tile, base_tile,
            err_msg=f"{r['tile_id']} differs from base {b['tile_id']}",
        )
        checked += 1
    assert checked > 0
