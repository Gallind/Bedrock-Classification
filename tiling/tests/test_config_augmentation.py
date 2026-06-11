# tiling/tests/test_config_augmentation.py
"""Tests for the augmentation section of the pipeline config."""
import pytest
from pydantic import ValidationError

from seabed_tiler.config import AugmentationConfig, AugPass, Config


def _minimal_config_dict(**overrides) -> dict:
    data = {
        "name": "testpoly",
        "src_dir": "DataBase/testpoly",
        "band_order": ["bathymetry"],
        "layers": [{"name": "bathymetry", "kind": "xyz", "path": "bathy.xyz"}],
        "labels": {"kind": "shapefile", "path": "labels.shp",
                   "classes": {"rock": 1, "sand": 3}},
    }
    data.update(overrides)
    return data


def test_config_without_augmentation_is_disabled():
    """Existing polygon YAMLs that lack the section must keep working."""
    cfg = Config(**_minimal_config_dict())
    assert cfg.augmentation.enabled is False
    assert cfg.augmentation.passes == []


def test_config_parses_augmentation_passes():
    cfg = Config(**_minimal_config_dict(augmentation={
        "enabled": True,
        "passes": [
            {"theta_offset_deg": -8.0, "u_shift_frac": 0.0, "v_shift_frac": 0.0},
            {"theta_offset_deg": 0.0, "u_shift_frac": 0.25, "v_shift_frac": 0.25},
        ],
    }))
    assert cfg.augmentation.enabled is True
    assert len(cfg.augmentation.passes) == 2
    assert cfg.augmentation.passes[0].theta_offset_deg == pytest.approx(-8.0)
    assert cfg.augmentation.passes[1].u_shift_frac == pytest.approx(0.25)


def test_aug_pass_defaults_are_identity():
    p = AugPass()
    assert p.theta_offset_deg == 0.0
    assert p.u_shift_frac == 0.0
    assert p.v_shift_frac == 0.0


@pytest.mark.parametrize("theta", [-46.0, 46.0, 90.0])
def test_aug_pass_rejects_extreme_theta(theta):
    with pytest.raises(ValidationError):
        AugPass(theta_offset_deg=theta)


@pytest.mark.parametrize("frac", [-0.1, 1.0, 2.5])
def test_aug_pass_rejects_shift_fraction_out_of_range(frac):
    with pytest.raises(ValidationError):
        AugPass(u_shift_frac=frac)
    with pytest.raises(ValidationError):
        AugPass(v_shift_frac=frac)


def test_augmentation_enabled_requires_at_least_one_pass():
    with pytest.raises(ValidationError):
        AugmentationConfig(enabled=True, passes=[])


def test_augmentation_rejects_duplicate_passes():
    """Two identical passes would write identical tiles twice."""
    p = {"theta_offset_deg": 5.0, "u_shift_frac": 0.25, "v_shift_frac": 0.0}
    with pytest.raises(ValidationError):
        AugmentationConfig(enabled=True, passes=[p, dict(p)])
