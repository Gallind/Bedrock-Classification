"""Shared fixtures: synthetic tiler-style run directories with tiny GeoTIFF pairs."""

import sys
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tiling" / "src"))

BANDS = ["backscatter", "bathymetry", "slope"]
NODATA = -9999.0
SIZE = 8


def write_tile_pair(run_dir: Path, tile_id: str, features: np.ndarray, label: np.ndarray):
    """Write one features/labels GeoTIFF pair the way the tiler lays them out."""
    feat_dir = run_dir / "tiles" / "features"
    lab_dir = run_dir / "tiles" / "labels"
    feat_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    transform = from_origin(600000, 3600000, 1.0, 1.0)
    h, w = label.shape
    with rasterio.open(
        feat_dir / f"{tile_id}.tif", "w", driver="GTiff", height=h, width=w,
        count=features.shape[0], dtype="float32", crs="EPSG:32636",
        transform=transform, nodata=NODATA,
    ) as dst:
        dst.write(features.astype("float32"))
        for i, band in enumerate(BANDS[: features.shape[0]], start=1):
            dst.set_band_description(i, band)
    with rasterio.open(
        lab_dir / f"{tile_id}.tif", "w", driver="GTiff", height=h, width=w,
        count=1, dtype="uint8", crs="EPSG:32636", transform=transform, nodata=0,
    ) as dst:
        dst.write(label.astype("uint8"), 1)


def make_run_dir(
    base_dir: Path,
    polygon: str,
    suffix: str,
    tiles: dict[str, tuple],
    theta_deg: float = 0.0,
    centers: dict[str, tuple] | None = None,
):
    """Create outputs/<polygon>/<run_tag><suffix>/ with tiles + manifest.csv.

    Manifest mirrors the real rotated manifests: theta_deg + center_x/center_y
    columns (defaults: the geometric center of the synthetic tile transform).
    """
    run_dir = base_dir / "outputs" / polygon / f"t128m_o50pct_r1m{suffix}"
    rows = ["tile_id,theta_deg,center_x,center_y,features_path,label_path"]
    for tile_id, (features, label) in tiles.items():
        write_tile_pair(run_dir, tile_id, features, label)
        cx, cy = (centers or {}).get(tile_id, (600000 + SIZE / 2, 3600000 - SIZE / 2))
        rel = run_dir.relative_to(base_dir)
        rows.append(
            f"{tile_id},{theta_deg},{cx},{cy},"
            f"{rel}/tiles/features/{tile_id}.tif,{rel}/tiles/labels/{tile_id}.tif"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.csv").write_text("\n".join(rows) + "\n")
    return run_dir


def make_tile(fill_per_band: list[float], label_value: int = 1):
    """Uniform (B,8,8) features + uniform (8,8) label."""
    features = np.stack(
        [np.full((SIZE, SIZE), v, dtype=np.float32) for v in fill_per_band]
    )
    label = np.full((SIZE, SIZE), label_value, dtype=np.uint8)
    return features, label


@pytest.fixture
def synth_dataset(tmp_path):
    """Three polygons, each with one base (_rot) tile and one augmented (_rotaug) tile."""
    for poly, base_fill in [("polygon1", 10.0), ("polygon3", 100.0), ("polygon4", 1000.0)]:
        fills = [base_fill, base_fill + 1, base_fill + 2]
        make_run_dir(tmp_path, poly, "_rot", {f"{poly}_base": make_tile(fills, label_value=1)})
        make_run_dir(tmp_path, poly, "_rotaug", {f"{poly}_aug": make_tile(fills, label_value=2)})
    return tmp_path
