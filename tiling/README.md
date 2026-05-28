# seabed_tiler

Splits a survey polygon's co-located layers — georeferenced raster renders
(`.jpg` + `.jgw` + `.prj`), point grids (`.xyz`), and a class shapefile — into equally
sized, **overlapping**, fully georeferenced GeoTIFF tiles for ML bedrock classification.

Every layer is resampled onto one common grid (UTM Zone 36N, configurable resolution)
and cut against the same world-coordinate windows, so each tile is a co-registered
**feature stack + class label** pair. Default tiles are **10 × 10 m with 50 % overlap**.

## Layout

```
tiling/
  config/
    default.yaml      # shared knobs (tile size, overlap, resolution, filters, output)
    polygon1.yaml     # per-polygon paths, layers, label class-map (overrides default)
  src/seabed_tiler/   # the package
  tests/              # grid math + label-name normalization
outputs/<name>/       # generated (gitignored): tiles/features, tiles/labels, manifests
```

## Setup

Requires GDAL-backed wheels. They install cleanly on the project's Python 3.14 `.venv`:

```bash
.venv/bin/python -m pip install -r tiling/requirements.txt
```

If a future environment has no wheels for `rasterio`/`geopandas`, create a Python 3.12
venv or a conda env and install there — do not force a source build.

## Run

```bash
PYTHONPATH=tiling/src .venv/bin/python -m seabed_tiler --config tiling/config/polygon1.yaml
```

Outputs land in `outputs/polygon1/`:

- `tiles/features/<name>_rRRR_cCCC.tif` — multiband float32 (bands per `band_order`).
- `tiles/labels/<name>_rRRR_cCCC.tif` — uint8 class id (`0` background, `1` rock,
  `2` shallow_rock, `3` sand).
- `manifest.csv` — one row per tile (bbox, valid fraction, per-class pixel counts, paths).
- `manifest.geojson` / `grid_preview.geojson` — drop into QGIS to inspect the 50 % overlap.

## Tuning (no code changes)

Edit the YAML and re-run:

- `tile_size_m`, `overlap` → tile geometry (stride = `tile_size_m * (1 - overlap)`).
- `target_resolution_m` → common grid resolution.
- `filters.min_valid_frac` → drop sparsely covered tiles (the `.xyz` surveys only cover
  ~half the area).
- `filters.require_label` → keep only tiles that contain labeled seabed.
- `extent` → `auto` (intersection of all layers) or explicit `[xmin, ymin, xmax, ymax]`.

## Adding another polygon

Copy `config/polygon1.yaml`, point `src_dir`/`path`s at the new folder, and adjust the
label `rules`. polygons 3–5 encode labels differently (separate shapefiles per class, or
class baked into raster filenames) and will each need a small adapter before they fit this
shapefile-based `labels` block.

## Tests

```bash
.venv/bin/python -m pytest tiling/tests -q
```
