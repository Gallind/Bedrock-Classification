"""crossval.py: fold construction and per-fold config overrides (no training)."""

import pytest

from seabed_unet.config import Config
from seabed_unet.crossval import fold_config, lopo_folds, summarize

POLYGONS = ["polygon1", "polygon3", "polygon4", "polygon5"]


def test_each_polygon_tested_exactly_once():
    folds = lopo_folds(POLYGONS)
    assert [f["test"][0] for f in folds] == POLYGONS


def test_fold_groups_are_disjoint_and_complete():
    for fold in lopo_folds(POLYGONS):
        groups = fold["train"] + fold["val"] + fold["test"]
        assert sorted(groups) == sorted(POLYGONS)
        assert len(set(groups)) == len(groups)


def test_test_polygon_never_in_its_own_train_or_val():
    for fold in lopo_folds(POLYGONS):
        assert fold["test"][0] not in fold["train"]
        assert fold["test"] != fold["val"]


def test_lopo_needs_three_polygons():
    with pytest.raises(ValueError, match=">= 3"):
        lopo_folds(["a", "b"])


def test_fold_config_switches_to_polygon_mode():
    cfg = Config(
        name="exp",
        bands=["slope"],
        split={"mode": "spatial_blocks", "polygons": POLYGONS},
    )
    fold = lopo_folds(POLYGONS)[2]  # test=polygon4
    fcfg = fold_config(cfg, fold, "exp_lopo")
    assert fcfg.split.mode == "polygon"
    assert fcfg.split.test == ["polygon4"]
    assert fcfg.name == "exp_lopo/fold_polygon4"
    assert "polygon4" not in fcfg.split.train
    # original config untouched
    assert cfg.split.mode == "spatial_blocks" and cfg.name == "exp"


def test_summarize_mean_std():
    reports = {
        "a": {"overall_accuracy": 0.4, "cohens_kappa": 0.1, "macro_dice": 0.5,
              "per_class": {"rock": {"dice": 0.6}}},
        "b": {"overall_accuracy": 0.6, "cohens_kappa": 0.3, "macro_dice": 0.7,
              "per_class": {"rock": {"dice": 0.8}}},
    }
    summary = summarize(reports, ["rock"])
    assert summary["overall_accuracy"]["mean"] == pytest.approx(0.5)
    assert summary["macro_dice"]["std"] == pytest.approx(0.1)
    assert summary["dice_rock"]["per_fold"] == {"a": 0.6, "b": 0.8}


def test_device_cycle_single_value():
    from seabed_unet.crossval import device_cycle
    assert device_cycle("cpu", 4) == ["cpu"] * 4
    assert device_cycle(None, 2) == ["cpu", "cpu"]


def test_device_cycle_comma_list_round_robin():
    from seabed_unet.crossval import device_cycle
    assert device_cycle("mps,cpu,cpu", 4) == ["mps", "cpu", "cpu", "mps"]
    assert device_cycle(" mps , cpu ", 3) == ["mps", "cpu", "mps"]


def test_threads_per_job_divides_physical_cores():
    from seabed_unet.crossval import threads_per_job
    assert threads_per_job(jobs=2, logical_cores=12) == 3   # 6 physical / 2
    assert threads_per_job(jobs=3, logical_cores=12) == 2
    assert threads_per_job(jobs=8, logical_cores=4) == 1    # never below 1


def test_fold_command_roundtrip():
    from seabed_unet.crossval import fold_command
    cmd = fold_command("cfg.yaml", "polygon4", "cpu", epochs=2, limit=8)
    assert "--fold" in cmd and "polygon4" in cmd
    assert cmd[cmd.index("--device") + 1] == "cpu"
    assert cmd[cmd.index("--epochs") + 1] == "2"
    assert cmd[cmd.index("--limit") + 1] == "8"
    # no smoke flags -> not in argv
    assert "--epochs" not in fold_command("c.yaml", "p", "cpu", None, None)
