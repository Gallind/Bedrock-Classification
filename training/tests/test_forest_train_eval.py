"""seabed_forest.train (+evaluate in Task 6): end-to-end on synthetic _rot run dirs."""

import json

import numpy as np
import yaml

from conftest import make_run_dir, write_tile_pair  # noqa: F401  (fixtures dir on sys.path)
from seabed_forest.config import load_forest_config
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
    # No augmented tiles exist, but the guard must hold regardless.
    assert summary["n_train_pixels"] == summary["n_train_pixels_base_only"]
