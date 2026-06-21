"""seabed_unet.export: headless recording of the multi-model tile-by-tile playback.

Records the tree lanes only (no U-Net checkpoints in the temp run dir, so those lanes
are skipped) and checks the static assets + manifest + catalog the website consumes.
"""

import json

import numpy as np
import yaml
from PIL import Image

from conftest import make_run_dir
from seabed_forest.config import load_forest_config
from seabed_forest.train import train_run


def _mixed_tile(fills, rng):
    feats = np.stack([np.full((8, 8), f, np.float32) + rng.normal(0, 0.01, (8, 8)).astype(np.float32)
                      for f in fills])
    label = np.empty((8, 8), np.uint8)
    label[:, 0:3] = 1
    label[:, 3:6] = 2
    label[:, 6:8] = 3
    label[0, :] = 0
    return feats, label


def _setup(tmp_path):
    rng = np.random.default_rng(0)
    for poly, base in [("polygon1", 1.0), ("polygon3", 5.0), ("polygon4", 9.0)]:
        tiles = {f"{poly}_a": _mixed_tile([base, base + 1, base + 2], rng)}
        if poly == "polygon4":  # a second tile so the lockstep loop has >1 step
            tiles[f"{poly}_b"] = _mixed_tile([base + 0.5, base + 1.5, base + 2.5], rng)
        make_run_dir(tmp_path, poly, "_rot", tiles)
    (tmp_path / "default.yaml").write_text(yaml.safe_dump({
        "outputs_dir": "outputs", "run_tag": "t128m_o50pct_r1m",
        "classes": {"rock": 1, "shallow_rock": 2, "sand": 3},
        "feature_nodata": -9999.0, "ignore_label": 0,
        "split": {"mode": "polygon", "train": ["polygon1"], "val": ["polygon3"],
                  "test": ["polygon4"], "use_augmented_for_train": False},
        "normalization": {"default_mode": "per_polygon",
                          "band_modes": {"bathymetry": "global", "slope": "global"}},
        "runs_dir": "runs",
    }))
    forest = tmp_path / "forest_3band.yaml"
    forest.write_text(yaml.safe_dump({
        "name": "forest_smoke", "bands": ["backscatter", "bathymetry", "slope"],
        "forest": {"models": ["random_forest", "hist_gradient_boosting"],
                   "spatial": {"enabled": True, "radius": 2}},
    }))
    (tmp_path / "experiment_3band.yaml").write_text(yaml.safe_dump({
        "name": "unet_3band", "bands": ["backscatter", "bathymetry", "slope"]}))
    (tmp_path / "experiment_2band.yaml").write_text(yaml.safe_dump({
        "name": "unet_2band", "bands": ["bathymetry", "slope"]}))
    return forest


def test_record_polygon_tree_lanes(tmp_path):
    forest_cfg_path = _setup(tmp_path)
    cfg, forest = load_forest_config(forest_cfg_path, base_dir=tmp_path)
    train_run(cfg, forest)  # writes runs/forest_smoke/model_*.joblib + normalization_stats.json

    from seabed_unet.export import record_polygon, rebuild_catalog

    data_dir = tmp_path / "webapp" / "data"
    unet_configs = {
        "unet_3band": tmp_path / "experiment_3band.yaml",
        "unet_2band": tmp_path / "experiment_2band.yaml",
    }
    manifest_path = record_polygon(
        tmp_path, "polygon4", forest_config=forest_cfg_path, unet_configs=unet_configs,
        data_dir=data_dir, max_long_side=1400,
    )
    m = json.loads(manifest_path.read_text())

    # tree lanes only (U-Net skipped — no checkpoints): raw + guided-spatial per tree kind
    lane_ids = {e["id"] for e in m["models"]}
    assert lane_ids == {"rf_raw", "rf_spatial", "hgb_raw", "hgb_spatial"}
    assert m["n_tiles"] == 2 and len(m["steps"]) == 2
    assert m["overlay_alpha"] == 0.8
    assert m["class_palette"]["1"] == [202, 0, 32]  # rock = red, from LABEL_COLORS

    session = manifest_path.parent
    assert (session / "backdrop.jpg").exists()
    truth = Image.open(session / "truth.png")
    assert truth.mode == "RGBA"
    assert truth.size == (m["map_size"]["width"], m["map_size"]["height"])

    step = m["steps"][0]
    assert len(step["outline_px"]) == 4
    assert set(step["bands"]) == {"backscatter", "bathymetry", "slope"}
    for band_rel in step["bands"].values():
        assert (session / band_rel).exists()
    for lane in lane_ids:
        entry = step["per_model"][lane]
        cmap = Image.open(session / entry["class_map"])
        assert cmap.mode == "RGBA" and cmap.size == truth.size
        assert (session / entry["pred"]).exists()

    # summaries are real numbers; raw vs guided-spatial differ in geometry, so allow either
    for e in m["models"]:
        assert e["summary"]["mean_dice"] is None or 0.0 <= e["summary"]["mean_dice"] <= 1.0
        assert set(e["summary"]["per_class_dice"]) == {"rock", "shallow_rock", "sand"}

    catalog_path = rebuild_catalog(data_dir)
    catalog = json.loads(catalog_path.read_text())
    assert [p["polygon"] for p in catalog["polygons"]] == ["polygon4"]
    assert {mm["id"] for mm in catalog["polygons"][0]["models"]} == lane_ids
    assert catalog["model_labels"]["rf_spatial"] == "Random Forest (guided)"


def test_models_filter_selects_subset(tmp_path):
    forest_cfg_path = _setup(tmp_path)
    cfg, forest = load_forest_config(forest_cfg_path, base_dir=tmp_path)
    train_run(cfg, forest)

    from seabed_unet.export import record_polygon

    data_dir = tmp_path / "webapp" / "data"
    unet_configs = {
        "unet_3band": tmp_path / "experiment_3band.yaml",
        "unet_2band": tmp_path / "experiment_2band.yaml",
    }
    manifest_path = record_polygon(
        tmp_path, "polygon4", forest_config=forest_cfg_path, unet_configs=unet_configs,
        data_dir=data_dir, models=["rf_raw"],
    )
    m = json.loads(manifest_path.read_text())
    assert {e["id"] for e in m["models"]} == {"rf_raw"}
