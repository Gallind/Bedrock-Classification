# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Israeli seabed classification pipeline for IOLR (Israel Oceanographic & Limnological Research). Multibeam echosounder (MBES) data from R/V Bat-Galim is processed into overlapping GeoTIFF tile pairs (features + labels) for training a U-Net seabed classifier (bedrock / shallow rock / sand).

Reference paper: Garone et al., Frontiers in Earth Science, 2023 — `DL_article.pdf`.

## Environment Setup

GDAL-backed Python wheels are required. Use the project `.venv`:

```powershell
.venv\Scripts\python -m pip install -r tiling\requirements.txt
```

If rasterio/geopandas wheels are unavailable for the current Python version, create a Python 3.12 venv or conda env — do not force a source build.

## Git LFS Setup (required for all contributors)

This repo tracks `.tif`, `.jpg`, and `.xyz` files via Git LFS. Without it you get pointer files instead of data.

```powershell
# Install once per machine
winget install GitHub.GitLFS

# Enable in your clone (run once after cloning)
git lfs install
git lfs pull
```

New `.tif`, `.jpg`, and `.xyz` files are handled automatically on `git add` — no extra steps.

Storage policy: only raw `DataBase/` inputs are versioned (via LFS). Generated `outputs/` are never committed — they are deterministic and every contributor regenerates them by re-running the tiler. Share frozen dataset releases via external storage, not git.

## Running the Tiler

```powershell
# Tile a polygon (outputs to outputs/<name>/<run-tag>/)
$env:PYTHONPATH="tiling\src"; .venv\Scripts\python -m seabed_tiler --config tiling\config\polygon1.yaml

# Rotation-aware tiles aligned to the annotation MBR (-> <run-tag>_rot/)
$env:PYTHONPATH="tiling\src"; .venv\Scripts\python -m seabed_tiler --config tiling\config\polygon1.yaml --rotated

# Deterministic augmentation passes (-> <run-tag>_rotaug/). Rules: docs/DATA_AUGMENTATION.md
$env:PYTHONPATH="tiling\src"; .venv\Scripts\python -m seabed_tiler --config tiling\config\polygon1.yaml --augment

# Convert tiles to viewable JPEGs
$env:PYTHONPATH="tiling\src"; .venv\Scripts\python -m seabed_tiler.to_jpg --tiles-dir outputs\polygon1

# Stitch tiles back for QA
$env:PYTHONPATH="tiling\src"; .venv\Scripts\python -m seabed_tiler.stitch --tiles-dir outputs\polygon1
```

## Tests

```powershell
.venv\Scripts\python -m pytest tiling\tests -q

# Single test
.venv\Scripts\python -m pytest tiling\tests\test_grid.py::test_stride_is_half_tile_gives_50pct_overlap -q
```

## Architecture

### Data

`DataBase/` holds four survey polygons (polygon1, polygon3, polygon4, polygon5). All raw data is in the repo: shapefiles in plain git, `.xyz` point grids and `.jpg` renders via Git LFS. Each polygon has co-located layers:
- **Bathymetry** — `.xyz` (X Y Z) or `.jpg` render
- **Backscatter** — `.xyz` or `.jpg` render
- **Slope** — `.xyz`
- **Labels** — shapefile(s) of annotated seabed class polygons

Classes: `rock=1`, `shallow_rock=2`, `sand=3`, `background=0` (unlabeled).

### `tiling/` Package

**Config layer** (`config.py`): Pydantic models (`Config`, `LayerConfig`, `LabelsConfig`). `load_config()` deep-merges `default.yaml` onto a polygon YAML. The `run_tag` property (`t128m_o50pct_r1m`) namespaces outputs so different tile geometries never collide.

**Alignment** (`align.py`): `build_grid_and_features()` computes the intersection extent of all layers, snaps the origin to `origin_snap_m`, reprojects everything to EPSG:32636 (UTM Zone 36N) at `target_resolution_m`, and returns a dict of aligned float32 arrays keyed by band name.

**Grid** (`grid.py`): Pure, I/O-free `build_windows()` returns `TileWindow` dataclasses in raster row/col order. Overlap is controlled by `stride_m = tile_size_m * (1 - overlap)`.

**Labels** (`labels.py`): Two rasterization strategies controlled by `labels.kind`:
- `shapefile` (polygon1): one file, class inferred from a NAME field via ordered regex rules (shallow before rock to avoid misclassification).
- `shapefile_per_class` (polygons 3/4/5): one shapefile per class, burned in priority order so rock wins on overlap. `polygonize: true` closes LineString rings before rasterizing.

**Tiler** (`tiler.py`): `run_tiling()` iterates windows, filters by `min_valid_frac` and `require_label`, writes co-registered `tiles/features/*.tif` (multiband float32) and `tiles/labels/*.tif` (uint8), and returns manifest rows.

**Outputs** per run-tag:
- `tiles/features/<name>_rRRR_cCCC.tif` — multiband float32, band descriptions set
- `tiles/labels/<name>_rRRR_cCCC.tif` — uint8 class ids
- `manifest.csv` — bbox, valid_frac, per-class pixel counts, file paths
- `manifest.geojson` / `grid_preview.geojson` — load in QGIS to inspect tiling

### Config Tuning (no code changes)

Edit YAML and re-run — `default.yaml` sets shared defaults, polygon YAMLs override:
- `tile_size_m`, `overlap` → tile geometry; stride derived as `size * (1 - overlap)`
- `target_resolution_m` → master grid resolution (default 1 m for U-Net standard)
- `filters.min_valid_frac` → drop sparsely covered tiles (XYZ surveys cover ~half the bbox)
- `filters.require_label` → keep only tiles with labeled seabed
- `extent` → `auto` (intersection of all layers) or explicit `[xmin, ymin, xmax, ymax]`

### Adding a New Polygon

Copy an existing polygon YAML, set `name`, `src_dir`, and update `layers` + `labels`. Polygons 3–5 use `kind: shapefile_per_class` with `class_files`. Polygon3 requires `polygonize: true` because its label shapefiles are LineString rings.

## Key Conventions

- CRS: EPSG:32636 (UTM Zone 36N, meters) throughout — matches survey `.prj` files.
- Nodata: `feature_nodata = -9999.0` (float32 layers), `label_nodata = 0` (uint8 labels).
- Valid pixel: a pixel is valid only where **every** feature band has real data (no nodata, no NaN).
- PYTHONPATH must include `tiling/src` for all module invocations (no install step).
- Augmentation: only rigid geometric transforms (D4 ops, rotated re-extraction passes). Photometric changes (brightness, depth offsets, zoom, noise) are prohibited — pixel values are physical measurements. Raw `DataBase/` bundles are read-only. See `docs/DATA_AUGMENTATION.md`.
- `.xyz` files are tracked via Git LFS like the other rasters — `git lfs pull` brings the complete raw dataset; nothing needs to be obtained out-of-band.
