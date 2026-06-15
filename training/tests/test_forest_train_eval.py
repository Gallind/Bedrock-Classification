"""seabed_forest.train (+evaluate in Task 6): end-to-end on synthetic _rot run dirs."""

import json

import numpy as np
import yaml

from conftest import make_run_dir, write_tile_pair  # noqa: F401  (fixtures dir on sys.path)
from seabed_forest.config import load_forest_config
from seabed_forest.evaluate import evaluate_run
from seabed_forest.train import train_run


def _mixed_tile(fills, rng):
    """(3,8,8) features + an (8,8) label with all three classes + some background."""
    feats = np.stack([np.full((8, 8), f, np.float32) + rng.normal(0, 0.01, (8, 8)).astype(np.float32)
                      for f in fills])
    label = np.empty((8, 8), np.uint8)
    label[:, 0:3] = 1
    label[:, 3:6] = 2
    label[:, 6:8] = 3
    label[0, :] = 0            # a background row (must be masked out)
    return feats, label


def _write_forest_yaml(tmp_path):
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
    exp = tmp_path / "forest_3band.yaml"
    exp.write_text(yaml.safe_dump({
        "name": "forest_smoke", "bands": ["backscatter", "bathymetry", "slope"],
        "forest": {"models": ["random_forest", "hist_gradient_boosting"], "dedup_overlap": True},
    }))
    return exp


def _build_synth(tmp_path):
    rng = np.random.default_rng(0)
    for poly, base in [("polygon1", 1.0), ("polygon3", 5.0), ("polygon4", 9.0)]:
        make_run_dir(tmp_path, poly, "_rot",
                     {f"{poly}_a": _mixed_tile([base, base + 1, base + 2], rng)})


def test_train_run_writes_artifacts(tmp_path):
    _build_synth(tmp_path)
    exp = _write_forest_yaml(tmp_path)
    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    summary = train_run(cfg, forest)

    run_dir = tmp_path / "runs" / "forest_smoke"
    assert (run_dir / "normalization_stats.json").exists()
    for kind in forest.models:
        assert (run_dir / f"model_{kind}.joblib").exists()
        assert (run_dir / f"feature_importance_{kind}.csv").exists()
    assert set(summary["models"]) == set(forest.models)
    assert summary["n_train_pixels"] > 0
    # No augmented tiles in this fixture, so all train tiles are base tiles.
    assert summary["n_base_tiles"] == summary["n_all_train_tiles"]


def test_rotaug_tiles_excluded_from_training(tmp_path):
    rng = np.random.default_rng(1)
    # polygon1 (train): one base _rot tile + one augmented _rotaug tile.
    make_run_dir(tmp_path, "polygon1", "_rot",
                 {"polygon1_base": _mixed_tile([1.0, 2.0, 3.0], rng)})
    make_run_dir(tmp_path, "polygon1", "_rotaug",
                 {"polygon1_aug": _mixed_tile([1.0, 2.0, 3.0], rng)})
    # val/test need base tiles too.
    make_run_dir(tmp_path, "polygon3", "_rot", {"polygon3_a": _mixed_tile([5.0, 6.0, 7.0], rng)})
    make_run_dir(tmp_path, "polygon4", "_rot", {"polygon4_a": _mixed_tile([9.0, 10.0, 11.0], rng)})

    (tmp_path / "default.yaml").write_text(yaml.safe_dump({
        "outputs_dir": "outputs", "run_tag": "t128m_o50pct_r1m",
        "classes": {"rock": 1, "shallow_rock": 2, "sand": 3},
        "feature_nodata": -9999.0, "ignore_label": 0,
        "split": {"mode": "polygon", "train": ["polygon1"], "val": ["polygon3"],
                  "test": ["polygon4"], "use_augmented_for_train": True},
        "normalization": {"default_mode": "per_polygon",
                          "band_modes": {"bathymetry": "global", "slope": "global"}},
        "runs_dir": "runs",
    }))
    exp = tmp_path / "forest_aug.yaml"
    exp.write_text(yaml.safe_dump({
        "name": "forest_aug_guard", "bands": ["backscatter", "bathymetry", "slope"],
        "forest": {"models": ["random_forest"], "dedup_overlap": False},
    }))

    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    summary = train_run(cfg, forest)

    # The augmented tile IS loaded into the train split (tile level)...
    assert summary["n_all_train_tiles"] == 2
    # ...but only the single BASE tile trains the forest.
    assert summary["n_base_tiles"] == 1
    # One 8x8 tile, 8 background pixels (top row=0) masked out -> 56 valid pixels.
    # dedup is off, so no further reduction. If the aug tile leaked in, this would be ~112.
    assert summary["n_train_pixels"] == 56


def test_evaluate_run_writes_metrics_and_comparison(tmp_path):
    _build_synth(tmp_path)
    exp = _write_forest_yaml(tmp_path)
    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    train_run(cfg, forest)
    reports = evaluate_run(cfg, forest, split="test")

    run_dir = tmp_path / "runs" / "forest_smoke"
    for kind in forest.models:
        mpath = run_dir / f"metrics_{kind}.json"
        assert mpath.exists()
        report = json.loads(mpath.read_text())
        assert "macro_dice" in report and "per_class" in report
        assert set(report["per_class"]) == {"rock", "shallow_rock", "sand"}
    assert (run_dir / "comparison.csv").exists()
    assert (run_dir / "comparison.md").exists()
    assert set(reports) == set(forest.models)


from seabed_forest.predict import majority_filter, predict_polygon_map


def test_majority_filter_smooths_singletons():
    import numpy as np
    cmap = np.ones((5, 5), np.uint8)
    cmap[2, 2] = 3                      # lone class-3 pixel in a sea of class 1
    out = majority_filter(cmap, 3, class_ids=[1, 2, 3])
    assert out[2, 2] == 1              # majority wins
    assert out.shape == cmap.shape


def test_predict_polygon_map_writes_outputs(tmp_path):
    import rasterio
    _build_synth(tmp_path)
    exp = _write_forest_yaml(tmp_path)
    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    train_run(cfg, forest)
    paths = predict_polygon_map(cfg, forest, polygon="polygon4")
    run_dir = tmp_path / "runs" / "forest_smoke"
    for kind in forest.models:
        tif = run_dir / "maps" / f"polygon4_pred_{kind}.tif"
        assert tif in paths and tif.exists()
        with rasterio.open(tif) as src:
            assert src.count == 1 and src.dtypes[0] == "uint8"
        assert (run_dir / "maps" / f"polygon4_pred_{kind}.jpg").exists()


from seabed_forest.crossval import run_lopo


def _write_blocks_yaml(tmp_path):
    (tmp_path / "default.yaml").write_text(yaml.safe_dump({
        "outputs_dir": "outputs", "run_tag": "t128m_o50pct_r1m",
        "classes": {"rock": 1, "shallow_rock": 2, "sand": 3},
        "feature_nodata": -9999.0, "ignore_label": 0,
        "split": {"mode": "spatial_blocks",
                  "polygons": ["polygon1", "polygon3", "polygon4"],
                  "use_augmented_for_train": False},
        "normalization": {"default_mode": "per_polygon",
                          "band_modes": {"bathymetry": "global", "slope": "global"}},
        "runs_dir": "runs",
    }))
    exp = tmp_path / "forest_3band.yaml"
    exp.write_text(yaml.safe_dump({
        "name": "forest_lopo_smoke", "bands": ["backscatter", "bathymetry", "slope"],
        "forest": {"models": ["random_forest"]},
    }))
    return exp


def test_run_lopo_writes_summary(tmp_path):
    _build_synth(tmp_path)
    exp = _write_blocks_yaml(tmp_path)
    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    summaries = run_lopo(cfg, forest)
    lopo_dir = tmp_path / "runs" / "forest_lopo_smoke_lopo"
    assert (lopo_dir / "summary_random_forest.json").exists()
    s = summaries["random_forest"]
    assert "macro_dice" in s and "mean" in s["macro_dice"]
    assert set(s["macro_dice"]["per_fold"]) == {"polygon1", "polygon3", "polygon4"}
