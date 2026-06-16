"""seabed_forest.watch: headless smoke of the multi-model live viewer."""

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


def test_run_watch_trees_only_spatial(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    exp = _setup(tmp_path)
    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    train_run(cfg, forest)
    from seabed_forest.watch import run_watch
    frames = run_watch(cfg, forest, polygon="polygon4", kinds=list(forest.models),
                       spatial=True, unet_cfg=None, delay=0.0, save=True, max_tiles=2, block=False)
    assert isinstance(frames, list) and len(frames) >= 1
    assert (tmp_path / "runs" / "forest_smoke" / "maps" / "polygon4_watch_multi.gif").exists()


def test_run_watch_no_spatial_single_model(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    exp = _setup(tmp_path)
    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    train_run(cfg, forest)
    from seabed_forest.watch import run_watch
    frames = run_watch(cfg, forest, polygon="polygon4", kinds=["random_forest"],
                       spatial=False, unet_cfg=None, delay=0.0, save=False, max_tiles=2, block=False)
    assert frames == []


def test_watch_families_heterogeneous_with_stub(tmp_path):
    """Exercise the generalized loop + 2-row layout with a stub non-spatial family
    standing in for the U-Net (no torch needed)."""
    import matplotlib
    matplotlib.use("Agg")
    from seabed_forest.watch import ModelFamily, _watch_families, build_tree_families
    from seabed_forest.predict import PosteriorMapAccumulator
    from seabed_unet.data import load_run_records
    from seabed_unet.normalize import load_stats
    from seabed_unet.predict import resolve_polygon_stats

    exp = _setup(tmp_path)
    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    train_run(cfg, forest)
    band_modes = cfg.normalization.modes_for(cfg.bands)
    recs = load_run_records(cfg.rot_dir("polygon4"), "polygon4", cfg.bands, cfg.base_dir, augmented=False)
    stats = resolve_polygon_stats(
        cfg, "polygon4", recs, load_stats(cfg.run_dir / "normalization_stats.json"), band_modes
    )
    families = build_tree_families(cfg, forest, list(forest.models), True, recs, stats)

    C = cfg.num_classes

    def stub_fields(r):
        return np.full((C,) + r.label.shape, 1.0 / C, dtype=np.float64), None

    families.append(ModelFamily(
        label="STUB",
        acc=PosteriorMapAccumulator(recs, cfg.class_ids, cfg.feature_nodata),
        tile_fields=stub_fields,
        supports_spatial=False,
    ))
    # 5 model maps (RF raw/spatial, HGB raw/spatial, STUB) + truth = 6 -> 3 cols x 2 rows
    frames = _watch_families(families, recs, cfg, forest, stats, band_modes,
                             polygon="polygon4", delay=0.0, save=True, block=False)
    assert isinstance(frames, list) and len(frames) >= 1
