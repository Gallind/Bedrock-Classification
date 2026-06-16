"""ForestConfig + loader. Reuses seabed_unet.config.Config for everything dataset-
related (split, normalization, bands, run dirs), so the forest's splits and
normalization are byte-identical to the U-Net's. The `forest:` YAML block is popped
before building Config (whose extra="forbid" would otherwise reject it)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from seabed_unet.config import Config, _deep_merge

SUPPORTED_MODELS = ("random_forest", "hist_gradient_boosting")


class RFParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_estimators: int = Field(default=300, ge=1)
    max_depth: int | None = None
    min_samples_leaf: int = Field(default=1, ge=1)
    n_jobs: int = -1
    class_weight: str = Field(default="balanced", pattern="^(balanced|balanced_subsample)$")

    @model_validator(mode="after")
    def _check_n_jobs(self) -> "RFParams":
        if self.n_jobs == 0:
            raise ValueError("n_jobs must be -1 (all CPUs) or >= 1; 0 is invalid for sklearn")
        return self


class HGBParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    learning_rate: float = Field(default=0.1, gt=0.0)
    max_iter: int = Field(default=300, ge=1)
    max_leaf_nodes: int = Field(default=31, ge=2)
    l2_regularization: float = Field(default=0.0, ge=0.0)
    early_stopping: bool = True


class SpatialConfig(BaseModel):
    """Edge-aware spatial regularization of the posterior (guided filter)."""
    model_config = ConfigDict(extra="forbid")
    enabled: bool = False
    method: str = Field(default="guided", pattern="^(guided)$")
    radius: int = Field(default=4, ge=1)          # box radius -> (2r+1) window
    eps: float = Field(default=1e-3, gt=0.0)      # guided-filter regularization (probs/guide in [0,1])
    guide_band: str = "bathymetry"                # feature band used as the edge guide


class ForestConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[str] = Field(default_factory=lambda: list(SUPPORTED_MODELS))
    seed: int = 42
    max_pixels_per_class: int | None = None   # None = all valid train pixels; else stratified cap
    dedup_overlap: bool = True                # collapse 50% tile overlap by world coordinate (train only)
    majority_filter_size: int = 0             # 0/1 = off; odd N>1 = NxN majority filter on prediction maps
    random_forest: RFParams = RFParams()
    hist_gradient_boosting: HGBParams = HGBParams()
    spatial: SpatialConfig = SpatialConfig()

    @model_validator(mode="after")
    def _check_fields(self) -> "ForestConfig":
        if not self.models:
            raise ValueError("forest.models must list at least one model")
        bad = [m for m in self.models if m not in SUPPORTED_MODELS]
        if bad:
            raise ValueError(f"unknown forest model(s) {bad}; use {list(SUPPORTED_MODELS)}")
        if len(set(self.models)) != len(self.models):
            raise ValueError(f"duplicate models in forest.models: {self.models}")
        if self.max_pixels_per_class is not None and self.max_pixels_per_class < 1:
            raise ValueError("max_pixels_per_class must be >= 1 or null")
        n = self.majority_filter_size
        if n not in (0, 1) and (n < 0 or n % 2 == 0):
            raise ValueError(f"majority_filter_size must be 0 or an odd integer >= 1, got {n}")
        return self


def load_forest_config(
    config_path: str | Path, base_dir: str | Path | None = None
) -> tuple[Config, ForestConfig]:
    """Load default.yaml (sibling) merged under the experiment config, then split it
    into a stock seabed_unet Config (dataset parts) and a ForestConfig (the `forest:` block)."""
    config_path = Path(config_path)
    base = Path(base_dir).resolve() if base_dir else Path.cwd()

    data: dict = {}
    default_path = config_path.parent / "default.yaml"
    if default_path.exists():
        data = yaml.safe_load(default_path.read_text()) or {}
    experiment = yaml.safe_load(config_path.read_text()) or {}
    merged = _deep_merge(data, experiment)

    forest_raw = merged.pop("forest", {})
    cfg = Config(**merged)
    cfg.base_dir = base
    return cfg, ForestConfig(**forest_raw)
