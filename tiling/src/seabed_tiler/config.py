"""Typed config loading: read default.yaml, deep-merge a polygon config on top, validate.

Everything the user tweaks lives in the YAML files under config/. This module turns them
into a validated ``Config`` object and resolves all paths relative to a base directory
(the repo root by default).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class LayerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    kind: Literal["raster_jpg", "xyz"]
    path: str
    to_gray: bool = False
    resampling: str = "nearest"
    native_res_m: float | None = None
    # JPEG rendering (used by to_jpg / stitch previews, not by the GeoTIFF tiles):
    cmap: str | None = None       # matplotlib colormap name; None -> grayscale
    hillshade: bool = False       # blend relief shading (for elevation-like layers)
    vert_exag: float = 5.0        # vertical exaggeration for hillshade


class LabelRule(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    pattern: str
    class_: str = Field(alias="class")


class LabelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["shapefile"]
    path: str
    name_field: str = "NAME"
    classes: dict[str, int]
    rules: list[LabelRule]


class FiltersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_valid_frac: float = 0.5
    require_label: bool = True


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dir: str = "outputs"
    features_dtype: str = "float32"
    label_nodata: int = 0
    feature_nodata: float = -9999.0
    compress: str = "deflate"
    write_preview_png: bool = False


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    src_dir: str
    crs: str = "EPSG:32636"
    target_resolution_m: float = 0.5
    tile_size_m: float = 10.0
    overlap: float = 0.5
    extent: str | list[float] = "auto"
    origin_snap_m: float = 10.0
    keep_partial_edge: bool = False
    band_order: list[str]
    layers: list[LayerConfig]
    labels: LabelsConfig
    filters: FiltersConfig = Field(default_factory=FiltersConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    # Set programmatically after load (not part of the YAML).
    base_dir: Path = Field(default_factory=Path.cwd, exclude=True)

    @property
    def stride_m(self) -> float:
        return self.tile_size_m * (1.0 - self.overlap)

    @property
    def src_path(self) -> Path:
        return self.base_dir / self.src_dir

    def layer_path(self, layer: LayerConfig) -> Path:
        return self.src_path / layer.path

    @property
    def labels_path(self) -> Path:
        return self.src_path / self.labels.path

    @property
    def out_dir(self) -> Path:
        return self.base_dir / self.output.dir / self.name


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
    """Load default.yaml (sibling of ``config_path``) merged under the polygon config."""
    config_path = Path(config_path)
    base = Path(base_dir).resolve() if base_dir else Path.cwd()

    data: dict = {}
    default_path = config_path.parent / "default.yaml"
    if default_path.exists():
        data = yaml.safe_load(default_path.read_text()) or {}

    polygon = yaml.safe_load(config_path.read_text()) or {}
    merged = _deep_merge(data, polygon)

    cfg = Config(**merged)
    cfg.base_dir = base
    return cfg


def validate_inputs(cfg: Config) -> None:
    """Fail fast if any declared input file (or shapefile companion) is missing."""
    missing: list[str] = []
    for layer in cfg.layers:
        if not cfg.layer_path(layer).exists():
            missing.append(str(cfg.layer_path(layer)))

    shp = cfg.labels_path
    for companion in (shp, shp.with_suffix(".shx"), shp.with_suffix(".dbf")):
        if not companion.exists():
            missing.append(str(companion))

    if missing:
        raise FileNotFoundError(
            "Missing input files:\n  " + "\n  ".join(missing)
        )
