# Developer Guide — Bedrock Classification Pipeline

## What This Project Is

This is a **data preparation pipeline** for training a deep learning seabed classifier.
It takes raw multibeam echosounder (MBES) survey data from the Israeli Mediterranean
Sea and converts it into co-registered GeoTIFF tile pairs (features + labels) that
a U-Net model can train on.

The scientific basis is Garone et al. (2023) — see `DL_article.pdf` for the paper.
The client is IOLR (Israel Oceanographic & Limnological Research), who collect the
data aboard the research vessel R/V Bat-Galim.

**What the pipeline does NOT do:** train the model. It stops at producing the training
dataset. Model training is a separate project that consumes the outputs of this one.

---

## What You Need To Know About The Domain

**Multibeam echosounder (MBES):** a hull-mounted sonar that sweeps a fan of acoustic
beams across the seabed and records two things per point:
- **Bathymetry (depth):** the Z value of each seabed point in meters
- **Backscatter:** how hard the acoustic signal bounced back (dB), correlated with
  sediment hardness

**Derived layers computed from bathymetry:**
- **Slope:** rate of depth change (degrees)
- **Hillshade:** artificial illumination for visual inspection

**Labels:** expert-drawn polygon shapefiles that annotate regions as one of:
- `0` — background (unlabeled / survey gap)
- `1` — rock (hard bedrock)
- `2` — shallow_rock
- `3` — sand

**Target output:** small overlapping image patches ("tiles") where each pixel is
exactly 1m x 1m, every feature band is co-registered to the same grid, and the
label tile shows the class at each pixel. This is the standard format for supervised
geospatial deep learning.

---

## Repository Layout

```
Bedrock-Classification/
├── DataBase/                 # Raw survey data (gitignored *.xyz, *.jpg files)
│   ├── polygon1/
│   ├── polygon3/
│   ├── polygon4/
│   └── polygon5/
│
├── tiling/                   # The core package (all pipeline code lives here)
│   ├── config/               # YAML configuration files
│   │   ├── default.yaml      # Shared knobs for all polygons
│   │   ├── polygon1.yaml     # Overrides specific to polygon 1
│   │   ├── polygon3.yaml
│   │   ├── polygon4.yaml
│   │   └── polygon5.yaml
│   ├── src/seabed_tiler/     # Python package
│   │   ├── __main__.py       # Entry point: python -m seabed_tiler
│   │   ├── config.py         # Pydantic config loading and validation
│   │   ├── align.py          # Builds master grid, reprojects all layers
│   │   ├── labels.py         # Rasterizes shapefile polygons to uint8 array
│   │   ├── grid.py           # Pure math: generates overlapping tile windows
│   │   ├── tiler.py          # Cuts tiles, filters, writes GeoTIFFs + manifest
│   │   ├── manifest.py       # Writes manifest.csv and manifest.geojson
│   │   ├── to_jpg.py         # Converts output tiles to viewable JPEGs
│   │   ├── stitch.py         # Stitches tiles back into full map (QA tool)
│   │   ├── viz.py            # Colormap, normalization, hillshade helpers
│   │   ├── io_utils.py       # GeoTIFF profile templates, tile_id naming
│   │   └── xyz.py            # Reads .xyz point files, snaps to grid
│   ├── tests/
│   │   ├── test_grid.py      # Unit tests for tile window math
│   │   └── test_labels.py    # Unit tests for label classification logic
│   └── requirements.txt
│
├── outputs/                  # Generated tiles (gitignored large TIFFs)
│   └── {polygon}/
│       └── {run_tag}/        # e.g. t128m_o50pct_r1m
│           ├── tiles/
│           │   ├── features/ # *.tif (multiband float32)
│           │   └── labels/   # *.tif (uint8 class ids)
│           ├── manifest.csv
│           ├── manifest.geojson
│           └── grid_preview.geojson
│
├── scripts/                  # Standalone visualization helpers (not part of pipeline)
│   ├── render_python.py      # Renders .xyz files to PNG with matplotlib
│   └── render_gmt.py         # Publication-quality rendering via PyGMT
│
├── DL_article.pdf            # Reference paper (Garone et al., 2023)
├── project-overview.md       # High-level requirements and background
└── DEVELOPER_GUIDE.md        # This file
```

---

## Setup

**Prerequisites:** Python 3.10+, Git, [Git LFS](https://git-lfs.github.com)

### Git LFS (required for all contributors)

This repo uses Git LFS to store large binary files (`.tif` tiles, `.jpg` renders, `.xyz`
point clouds). Without LFS installed, you will get small text pointer files instead of the
actual data.

```powershell
# Install Git LFS once per machine (if not already installed)
winget install GitHub.GitLFS

# Enable LFS in your local clone (run once after cloning)
git lfs install

# Pull all LFS-tracked files
git lfs pull
```

When you commit new `.tif`, `.jpg`, or `.xyz` files, Git LFS handles them automatically —
no extra steps needed beyond the normal `git add` / `git commit` workflow.

### Python environment

```powershell
# Clone and enter the repo
cd c:\Dev\Bedrock-Classification

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r tiling/requirements.txt
```

**Raw data files** (not in git — large binary/point files) must be placed in `DataBase/`
following the per-polygon structure described in each YAML config under `tiling/config/`.

---

## Running The Pipeline

The pipeline is invoked per polygon. Each run produces a fully self-contained output
folder tagged with the tile geometry (e.g. `t128m_o50pct_r1m`).

```powershell
# Set PYTHONPATH so the package is importable without pip install
$env:PYTHONPATH = "tiling/src"

# Run tiling for polygon1
.venv\Scripts\python -m seabed_tiler --config tiling/config/polygon1.yaml

# Run for other polygons
.venv\Scripts\python -m seabed_tiler --config tiling/config/polygon3.yaml
.venv\Scripts\python -m seabed_tiler --config tiling/config/polygon4.yaml
.venv\Scripts\python -m seabed_tiler --config tiling/config/polygon5.yaml
```

The run_tag in the output folder name encodes the key parameters:
- `t128m` — 128m tile size
- `o50pct` — 50% overlap between adjacent tiles
- `r1m` — 1m target resolution per pixel

**After tiling, convert tiles to viewable JPEGs:**

```powershell
.venv\Scripts\python -m seabed_tiler.to_jpg --tiles-dir outputs/polygon1 --what both
```

**Stitch tiles back into full map (QA check):**

```powershell
.venv\Scripts\python -m seabed_tiler.stitch --tiles-dir outputs/polygon1
```

---

## Running Tests

```powershell
$env:PYTHONPATH = "tiling/src"
.venv\Scripts\python -m pytest tiling/tests/ -v
```

Tests cover:
- `test_grid.py` — tile window math (overlap, stride, edge handling)
- `test_labels.py` — shapefile classification rules, geometry repair

---

## Pipeline Stages In Detail

Data flows through six sequential stages, each in its own module:

### 1. Config (`config.py`)
Loads `default.yaml` + the polygon YAML and deep-merges them (polygon values win).
Validates that all referenced input files exist. Builds the `run_tag` string.

Key Pydantic models: `Config`, `LayerConfig`, `LabelsConfig`.

### 2. Alignment (`align.py`)
Takes all input layers (different formats, resolutions, CRS) and puts them on a
single shared pixel grid:
- Reprojects everything to **EPSG:32636** (UTM Zone 36N, meters)
- Clips to the intersection of all layer extents (only pixels covered by all bands)
- Snaps grid origin to 10m multiples for reproducibility across runs
- Resamples JPEG rasters via `rasterio.warp`
- Snaps XYZ point clouds to grid cells (gaps become `nodata = -9999.0`)

Output: a dict with the master `Affine` transform, shape, and one float32 array
per feature band.

### 3. Label Rasterization (`labels.py`)
Burns shapefile polygons onto the master grid. Two strategies controlled by config:

- **`shapefile`** (polygon1): single file, class id extracted from a NAME field
  via ordered regex rules. "shallow rock" must be matched before "rock" to avoid
  the shorter pattern eating the longer one.

- **`shapefile_per_class`** (polygons 3-5): one shapefile per class, burned in
  priority order. Also handles the case where polygons were digitized as LineString
  rings (polygon3) by closing and polygonizing them.

Output: uint8 array with class ids (0/1/2/3).

### 4. Tile Window Generation (`grid.py`)
Pure math, no I/O. Given the extent and tile parameters, returns every
`TileWindow(row, col, xmin, ymin, xmax, ymax)` in row-major order.

`stride = tile_size_m * (1 - overlap_fraction)`

Rows increase southward (row 0 at the top/ymax). This is entirely unit-tested
and has no side effects.

### 5. Tiling + Filtering (`tiler.py`)
The core loop. For each window:
1. Extracts the multiband feature patch and label patch
2. Computes `valid_frac` = fraction of pixels where ALL feature bands are real data
3. Drops the tile if `valid_frac < 0.5` (sparse survey coverage)
4. Optionally drops if no label pixels present (`require_label = true`)
5. Writes two co-registered GeoTIFFs: `features/*.tif` and `labels/*.tif`
6. Records a manifest row

**Tile filename:** `{polygon_name}_r{row:03d}_c{col:03d}.tif`

### 6. Manifest (`manifest.py`)
Writes three files to the run output directory:
- `manifest.csv` — one row per written tile: bbox, valid_frac, per-class pixel counts,
  relative paths to feature and label TIFFs
- `manifest.geojson` — same data in WGS84 for loading in QGIS
- `grid_preview.geojson` — ALL candidate windows (pre-filter) so you can visualize
  the full 50%-overlap tiling grid

---

## Configuration Reference

Key fields in `default.yaml` (all overridable per polygon):

| Field | Default | Meaning |
|---|---|---|
| `tile_size_m` | 128 | Tile width and height in meters |
| `overlap` | 0.5 | Fraction of overlap (0.5 = 50%) |
| `target_resolution_m` | 1 | Pixel size in meters |
| `crs` | EPSG:32636 | Output CRS (UTM Zone 36N) |
| `min_valid_frac` | 0.5 | Minimum fraction of real pixels to keep tile |
| `require_label` | true | Drop tiles with no labeled pixels |
| `origin_snap_m` | 10 | Grid origin snapped to this multiple |

Per-polygon YAMLs also specify:
- `layers` — band definitions (path, format, resampling method, visualization)
- `labels` — shapefile path(s), strategy, class rules, priority order
- `band_order` — which bands to include and in what order in the feature TIF

---

## The Four Polygons

| Polygon | Notes |
|---|---|
| polygon1 | Reference polygon. Single label shapefile with all 3 classes. Cleanest data. |
| polygon3 | Labels are per-class shapefiles. Geometry was digitized as LineString rings — pipeline auto-closes and polygonizes them. |
| polygon4 | Per-class shapefiles. Standard polygon geometry. |
| polygon5 | Per-class shapefiles. Standard polygon geometry. |

---

## Output Format For Model Training

Each tile is a pair of co-registered GeoTIFFs:

**Features TIF** — multiband float32
- Band count and order defined by `band_order` in config
- Nodata value: `-9999.0`
- Values are raw physical units (meters depth, dB backscatter, degrees slope)
- The model training code is responsible for normalization to [0, 1]

**Labels TIF** — single band uint8
- Pixel values: 0=background, 1=rock, 2=shallow_rock, 3=sand
- Nodata value: 255 (outside all label polygons)
- Spatial extent and resolution are identical to the paired features TIF

---

## Key Design Decisions

**Why deep-merged YAML configs?**
The four polygons share 90% of their settings. Keeping a single `default.yaml`
means changes to shared settings only need to happen once.

**Why per-run-tag output directories?**
Running with different tile sizes or overlaps never overwrites previous outputs.
Researchers can compare `t10m_o50pct_r1m` vs `t128m_o50pct_r1m` side by side.

**Why nodata = -9999.0 for sparse XYZ surveys?**
MBES surveys have gaps (turns, equipment gaps, depth limits). Forcing those cells
to nodata makes `valid_frac` filtering honest — the model will not see pixels it
has no data for.

**Why is grid.py pure math with no I/O?**
It makes the tile windowing logic fully unit-testable without touching the filesystem
or requiring any geospatial data.

**Why EPSG:32636 throughout?**
That is the native CRS of the Israeli survey .prj files. Staying in it preserves
the original meter-scale accuracy and avoids double reprojection artifacts.

---

## Adding A New Polygon

1. Place raw data files in `DataBase/{polygon_name}/`
2. Copy `tiling/config/polygon1.yaml` to `tiling/config/{polygon_name}.yaml`
3. Edit the new YAML to point `layers` and `labels` at your actual files
4. If your labels are per-class shapefiles, set `strategy: shapefile_per_class`
5. Run `python -m seabed_tiler --config tiling/config/{polygon_name}.yaml`
6. Inspect `outputs/{polygon_name}/{run_tag}/manifest.geojson` in QGIS to verify

---

## Common Issues

**`FileNotFoundError` on a layer path**
The config validator checks all paths at startup. Edit the YAML to point at the
correct relative path from the project root.

**Tiles are all dropped (0 written)**
Check `min_valid_frac` — if survey coverage is very sparse, lower it to 0.3.
Also check that the label shapefile CRS matches `crs` in config (or that the
pipeline can reproject it — it reprojects automatically if EPSG codes differ).

**Labels all burn as class 0 (background)**
For `shapefile` strategy: check that the regex rules in `labels.classification_rules`
match the actual strings in your shapefile NAME field. Enable debug logging to see
what values are being matched.

**XYZ file reads slowly**
Very large `.xyz` point clouds (tens of millions of points) are the main bottleneck.
The `xyz.py` reader uses pandas for vectorized snapping — ensure you have enough RAM.
