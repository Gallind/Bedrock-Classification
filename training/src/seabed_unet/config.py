"""Experiment configuration: pydantic models + YAML deep-merge loading.

Mirrors seabed_tiler.config: default.yaml (sibling of the experiment config) is
deep-merged UNDER the experiment YAML, and every model forbids unknown keys so
config typos fail fast.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class SplitConfig(BaseModel):
    """Spatial split (random tile splits are invalid: 50% overlap).

    mode=polygon: whole-polygon holdout via train/val/test lists.
    mode=spatial_blocks: all `polygons` contribute; each is cut into contiguous
    train/val/test regions along its survey axis (`fractions`), with tiles
    within `buffer_m` of a boundary dropped so splits share zero pixels.
    """

    model_config = ConfigDict(extra="forbid")

    mode: str = Field(default="polygon", pattern="^(polygon|spatial_blocks)$")
    # polygon mode
    train: list[str] = Field(default_factory=list)
    val: list[str] = Field(default_factory=list)
    test: list[str] = Field(default_factory=list)
    # spatial_blocks mode
    polygons: list[str] = Field(default_factory=list)
    fractions: tuple[float, float, float] = (0.7, 0.15, 0.15)
    # 96 m > tile half-diagonal (90.5 m) => kept tiles in different splits are
    # >= 192 m apart, beyond the 181.1 m worst-case overlap distance.
    buffer_m: float = Field(default=96.0, ge=0.0)
    use_augmented_for_train: bool = True

    @model_validator(mode="after")
    def _check_mode_fields(self) -> "SplitConfig":
        if self.mode == "polygon":
            for group_name in ("train", "val", "test"):
                if not getattr(self, group_name):
                    raise ValueError(
                        f"split.{group_name} must list at least one polygon (mode=polygon)"
                    )
            all_polys = self.train + self.val + self.test
            dupes = {p for p in all_polys if all_polys.count(p) > 1}
            if dupes:
                raise ValueError(f"polygon(s) in more than one split: {sorted(dupes)}")
        else:  # spatial_blocks
            if not self.polygons:
                raise ValueError("split.polygons must be non-empty (mode=spatial_blocks)")
            if len(set(self.polygons)) != len(self.polygons):
                raise ValueError(f"duplicate polygons in split.polygons: {self.polygons}")
            if any(f <= 0 for f in self.fractions) or abs(sum(self.fractions) - 1.0) > 1e-6:
                raise ValueError(
                    f"split.fractions must be positive and sum to 1, got {self.fractions}"
                )
        return self


class NormalizationConfig(BaseModel):
    """Per-band normalization modes.

    per_polygon = survey self-normalizes (required for backscatter: units differ
    per survey); global = one train-only range applied everywhere (preserves
    absolute depth/slope across surveys). Bands absent from band_modes fall back
    to default_mode.
    """

    model_config = ConfigDict(extra="forbid")

    default_mode: str = Field(default="per_polygon", pattern="^(per_polygon|global)$")
    band_modes: dict[str, str] = Field(default_factory=dict)
    clip_percentiles: tuple[float, float] = (2.0, 98.0)

    @model_validator(mode="after")
    def _check_fields(self) -> "NormalizationConfig":
        lo, hi = self.clip_percentiles
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError(f"clip_percentiles must satisfy 0 <= lo < hi <= 100, got {self.clip_percentiles}")
        bad = {b: m for b, m in self.band_modes.items() if m not in ("per_polygon", "global")}
        if bad:
            raise ValueError(f"invalid band_modes (use per_polygon|global): {bad}")
        return self

    def modes_for(self, bands: list[str]) -> dict[str, str]:
        """Fully-resolved {band: mode} for the experiment's band list."""
        return {b: self.band_modes.get(b, self.default_mode) for b in bands}


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_filters: int = Field(default=16, ge=4, le=128)
    depth: int = Field(default=4, ge=2, le=5)


class LossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ce_weight: float = Field(default=0.5, ge=0.0)
    dice_weight: float = Field(default=0.5, ge=0.0)
    class_weights: str = Field(default="auto", pattern="^(auto|none)$")

    @model_validator(mode="after")
    def _check_nonzero(self) -> "LossConfig":
        if self.ce_weight + self.dice_weight <= 0.0:
            raise ValueError("ce_weight + dice_weight must be > 0")
        return self


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    batch_size: int = Field(default=16, ge=1)
    epochs: int = Field(default=200, ge=1)
    lr: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    scheduler_patience: int = Field(default=10, ge=1)
    scheduler_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    early_stop_patience: int = Field(default=25, ge=1)
    d4_augment: bool = True
    device: str = "auto"


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    bands: list[str] = Field(min_length=1)
    outputs_dir: str = "outputs"
    run_tag: str = "t128m_o50pct_r1m"
    classes: dict[str, int] = Field(default={"rock": 1, "shallow_rock": 2, "sand": 3})
    feature_nodata: float = -9999.0
    ignore_label: int = 0
    split: SplitConfig
    normalization: NormalizationConfig = NormalizationConfig()
    model: ModelConfig = ModelConfig()
    loss: LossConfig = LossConfig()
    train: TrainConfig = TrainConfig()
    runs_dir: str = "training/runs"
    base_dir: Path = Path(".")

    @model_validator(mode="after")
    def _check_classes(self) -> "Config":
        ids = sorted(self.classes.values())
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate class ids: {self.classes}")
        if self.ignore_label in ids:
            raise ValueError(f"ignore_label {self.ignore_label} collides with a class id")
        return self

    # --- derived ---

    @property
    def class_ids(self) -> list[int]:
        """Label-raster ids in model-channel order (channel i <-> class_ids[i])."""
        return sorted(self.classes.values())

    @property
    def id_to_name(self) -> dict[int, str]:
        return {v: k for k, v in self.classes.items()}

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    @property
    def run_dir(self) -> Path:
        return self.base_dir / self.runs_dir / self.name

    def rot_dir(self, polygon: str) -> Path:
        return self.base_dir / self.outputs_dir / polygon / f"{self.run_tag}_rot"

    def rotaug_dir(self, polygon: str) -> Path:
        return self.base_dir / self.outputs_dir / polygon / f"{self.run_tag}_rotaug"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto ``base`` (override wins on scalar conflicts)."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(config_path: str | Path, base_dir: str | Path | None = None) -> Config:
    """Load default.yaml (sibling of ``config_path``) merged under the experiment config."""
    config_path = Path(config_path)
    base = Path(base_dir).resolve() if base_dir else Path.cwd()

    data: dict = {}
    default_path = config_path.parent / "default.yaml"
    if default_path.exists():
        data = yaml.safe_load(default_path.read_text()) or {}

    experiment = yaml.safe_load(config_path.read_text()) or {}
    merged = _deep_merge(data, experiment)

    cfg = Config(**merged)
    cfg.base_dir = base
    return cfg
