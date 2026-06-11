"""Config layer: YAML deep-merge, validation, derived paths."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seabed_unet.config import Config, SplitConfig, load_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"

BASE_KWARGS = dict(
    name="t",
    bands=["bathymetry", "slope"],
    split={"train": ["polygon1"], "val": ["polygon3"], "test": ["polygon4"]},
)


def test_real_experiment_configs_load():
    for exp, n_bands in [("experiment_3band.yaml", 3), ("experiment_2band.yaml", 2)]:
        cfg = load_config(CONFIG_DIR / exp, base_dir="/repo")
        assert len(cfg.bands) == n_bands
        assert cfg.split.train == ["polygon1", "polygon5"]
        assert cfg.split.val == ["polygon3"]
        assert cfg.split.test == ["polygon4"]
        assert cfg.num_classes == 3
        assert cfg.class_ids == [1, 2, 3]


def test_experiment_overrides_default_via_deep_merge(tmp_path):
    (tmp_path / "default.yaml").write_text(
        "run_tag: tag0\ntrain: {seed: 1, batch_size: 4}\n"
        "split: {train: [polygon1], val: [polygon3], test: [polygon4]}\n"
    )
    (tmp_path / "exp.yaml").write_text(
        "name: e\nbands: [slope]\ntrain: {seed: 7}\n"
    )
    cfg = load_config(tmp_path / "exp.yaml")
    assert cfg.train.seed == 7          # overridden
    assert cfg.train.batch_size == 4    # inherited through deep-merge
    assert cfg.run_tag == "tag0"


def test_derived_paths():
    cfg = Config(**BASE_KWARGS)
    cfg.base_dir = Path("/repo")
    assert cfg.rot_dir("polygon4") == Path("/repo/outputs/polygon4/t128m_o50pct_r1m_rot")
    assert cfg.rotaug_dir("polygon1") == Path("/repo/outputs/polygon1/t128m_o50pct_r1m_rotaug")
    assert cfg.run_dir == Path("/repo/training/runs/t")


def test_split_rejects_polygon_in_two_groups():
    with pytest.raises(ValueError, match="more than one split"):
        SplitConfig(train=["polygon1"], val=["polygon1"], test=["polygon4"])


def test_split_rejects_empty_group():
    with pytest.raises(ValueError, match="at least one polygon"):
        SplitConfig(train=["polygon1"], val=[], test=["polygon4"])


def test_unknown_key_fails_fast():
    with pytest.raises(Exception, match="tile_sizee"):
        Config(**BASE_KWARGS, tile_sizee=128)


def test_ignore_label_cannot_be_a_class():
    with pytest.raises(ValueError, match="collides"):
        Config(**{**BASE_KWARGS, "classes": {"rock": 0}})
