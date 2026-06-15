"""seabed_forest.watch: headless smoke of the live RF/HGB raw-vs-spatial viewer."""

import numpy as np
import yaml

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
        make_run_dir(tmp_path, poly, "_rot", {f"{poly}_a": _mixed_tile([base, base + 1, base + 2], rng)})
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
        "forest": {"models": ["random_forest", "hist_gradient_boosting"],
                   "spatial": {"enabled": True, "radius": 2}},
    }))
    return exp


def test_run_watch_headless_both_models_spatial(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    exp = _setup(tmp_path)
    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    train_run(cfg, forest)
    from seabed_forest.watch import run_watch
    frames = run_watch(cfg, forest, polygon="polygon4", kinds=list(forest.models),
                       spatial=True, delay=0.0, save=True, max_tiles=2, block=False)
    assert isinstance(frames, list) and len(frames) >= 1
    # GIF written next to the run's maps
    assert (tmp_path / "runs" / "forest_smoke" / "maps" / "polygon4_watch_forest.gif").exists()


def test_run_watch_headless_no_spatial_single_model(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    exp = _setup(tmp_path)
    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    train_run(cfg, forest)
    from seabed_forest.watch import run_watch
    frames = run_watch(cfg, forest, polygon="polygon4", kinds=["random_forest"],
                       spatial=False, delay=0.0, save=False, max_tiles=2, block=False)
    assert frames == []   # save=False -> no frames captured
