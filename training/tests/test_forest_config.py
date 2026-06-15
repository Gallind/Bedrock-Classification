"""seabed_forest.config: ForestConfig validation + load_forest_config (pops `forest:`)."""

import pytest
import yaml

from seabed_forest.config import ForestConfig, load_forest_config


def test_forest_config_defaults():
    fc = ForestConfig()
    assert fc.models == ["random_forest", "hist_gradient_boosting"]
    assert fc.dedup_overlap is True
    assert fc.majority_filter_size == 0
    assert fc.random_forest.n_estimators == 300
    assert fc.hist_gradient_boosting.max_iter == 300


def test_forest_config_rejects_unknown_model():
    with pytest.raises(ValueError, match="unknown forest model"):
        ForestConfig(models=["random_forest", "svm"])


def test_forest_config_rejects_even_filter():
    with pytest.raises(ValueError, match="majority_filter_size"):
        ForestConfig(majority_filter_size=4)


def test_forest_config_rejects_unknown_key():
    with pytest.raises(Exception, match="bogus"):
        ForestConfig(bogus=1)


def test_forest_config_rejects_empty_models():
    with pytest.raises(ValueError, match="at least one model"):
        ForestConfig(models=[])


def test_forest_config_rejects_duplicate_models():
    with pytest.raises(ValueError, match="duplicate models"):
        ForestConfig(models=["random_forest", "random_forest"])


def test_forest_config_rejects_zero_max_pixels():
    with pytest.raises(ValueError, match="max_pixels_per_class"):
        ForestConfig(max_pixels_per_class=0)


def test_forest_config_rejects_zero_n_jobs():
    from seabed_forest.config import RFParams
    with pytest.raises(ValueError, match="n_jobs"):
        RFParams(n_jobs=0)


def test_load_forest_config_pops_forest_block(tmp_path):
    (tmp_path / "default.yaml").write_text(yaml.safe_dump({
        "run_tag": "t128m_o50pct_r1m",
        "classes": {"rock": 1, "shallow_rock": 2, "sand": 3},
        "feature_nodata": -9999.0, "ignore_label": 0,
        "split": {"mode": "spatial_blocks", "polygons": ["polygon1", "polygon3", "polygon4"],
                  "use_augmented_for_train": False},
        "normalization": {"default_mode": "per_polygon",
                          "band_modes": {"bathymetry": "global", "slope": "global"}},
    }))
    exp = tmp_path / "forest_3band.yaml"
    exp.write_text(yaml.safe_dump({
        "name": "forest_3band", "bands": ["backscatter", "bathymetry", "slope"],
        "forest": {"models": ["random_forest"], "max_pixels_per_class": 5000},
    }))
    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    assert cfg.name == "forest_3band"
    assert cfg.bands == ["backscatter", "bathymetry", "slope"]
    assert cfg.split.use_augmented_for_train is False  # forest never trains on _rotaug
    assert forest.models == ["random_forest"]
    assert forest.max_pixels_per_class == 5000


def test_spatial_config_defaults():
    fc = ForestConfig()
    assert fc.spatial.enabled is False
    assert fc.spatial.method == "guided"
    assert fc.spatial.radius == 4
    assert fc.spatial.eps == 0.001
    assert fc.spatial.guide_band == "bathymetry"


def test_spatial_config_rejects_unknown_method():
    from seabed_forest.config import SpatialConfig
    with pytest.raises(ValueError):
        SpatialConfig(method="crf")


def test_spatial_config_rejects_nonpositive_radius():
    from seabed_forest.config import SpatialConfig
    with pytest.raises(ValueError):
        SpatialConfig(radius=0)


def test_spatial_config_rejects_unknown_key():
    from seabed_forest.config import SpatialConfig
    with pytest.raises(Exception):
        SpatialConfig(bogus=1)


def test_forest_config_loads_spatial_block(tmp_path):
    import yaml
    (tmp_path / "default.yaml").write_text(yaml.safe_dump({
        "run_tag": "t128m_o50pct_r1m",
        "classes": {"rock": 1, "shallow_rock": 2, "sand": 3},
        "feature_nodata": -9999.0, "ignore_label": 0,
        "split": {"mode": "spatial_blocks", "polygons": ["polygon1", "polygon3", "polygon4"],
                  "use_augmented_for_train": False},
        "normalization": {"default_mode": "per_polygon",
                          "band_modes": {"bathymetry": "global", "slope": "global"}},
    }))
    exp = tmp_path / "forest_3band.yaml"
    exp.write_text(yaml.safe_dump({
        "name": "forest_3band", "bands": ["backscatter", "bathymetry", "slope"],
        "forest": {"models": ["random_forest"],
                   "spatial": {"enabled": True, "radius": 6, "eps": 0.01, "guide_band": "backscatter"}},
    }))
    cfg, forest = load_forest_config(exp, base_dir=tmp_path)
    assert forest.spatial.enabled is True
    assert forest.spatial.radius == 6
    assert forest.spatial.guide_band == "backscatter"
