# Data Augmentation Contract

This document is the binding contract for how the seabed tile dataset may be
expanded and how any consumer (the future U-Net training project) must treat
augmented tiles. It exists because every transform we apply to this data risks
silently corrupting physical measurements or leaking test data into training.

## Why the raw bundles are never touched

Each survey area in `DataBase/` is a bundle of coupled files:

| File | Role |
|---|---|
| `*.jpg` | rendered raster (e.g. backscatter) -- pixel values only |
| `*.jgw` | world file: maps the jpg's pixels to UTM coordinates |
| `*.prj` | coordinate reference system definition |
| `*.shp` / `*.shx` / `*.dbf` | label polygons in UTM coordinates |
| `*.xyz` | measured point grids (bathymetry, slope) in UTM coordinates |

These files reference each other only through georeferencing. If you rotate or
flip the `.jpg` without rewriting the `.jgw`, or move imagery without moving the
label polygons, **nothing fails** -- the pipeline reprojects the now-inconsistent
layers without error and produces plausible-looking tiles whose features and
labels disagree. The model then trains on label noise that is invisible until
evaluation. For this reason:

- Raw bundles are read-only inputs. No augmentation, ever, at the bundle level.
- All augmentation happens downstream of `align.py`, where every layer has been
  fused onto one master grid and features + labels are guaranteed to transform
  together.

The same principle is encoded in the tile-level API: `seabed_tiler.augment`
transforms the (features, label) pair as one atomic unit and deliberately offers
no single-array transform.

## Prohibited transforms (and why)

Pixel values are physical measurements: backscatter intensity (sediment hardness
proxy), bathymetry (meters), slope (degrees). The following are prohibited:

| Transform | Why it corrupts the data |
|---|---|
| Brightness / contrast / gamma | Backscatter intensity encodes sediment hardness; changing it changes the apparent seabed type |
| Depth offsets or scaling | Class 2 is *shallow* rock -- depth correlates with class; shifted depths teach wrong physics |
| Zoom / rescale | A pixel is 1 m x 1 m by contract; zooming fabricates resolutions that do not exist |
| Elastic / non-rigid warps | Distorts geological structure and breaks bathymetry-slope consistency (slope is derived from bathymetry) |
| Per-band noise | Bands must stay mutually consistent; noisy bathymetry with clean slope is physically impossible |
| Color jitter / channel shuffling | Bands are distinct physical quantities, not interchangeable color channels |

## Allowed transforms

Only rigid geometric transforms, applied identically to all feature bands and the
label tile:

1. **Source-level re-extraction passes** (`--augment`, this pipeline): the rotated
   tile grid is re-extracted from the master grid with the MBR angle shifted by a
   few degrees and/or the grid origin shifted by a fraction of the stride. Each
   pass is a genuine resampling of the source data. Passes are deterministic and
   listed explicitly in `tiling/config/default.yaml` under `augmentation.passes`,
   so every re-run produces identical tiles. Features are warped bilinear, labels
   nearest (class ids never interpolate).
2. **Training-time D4 transforms** (`seabed_tiler/augment.py`): the 8 ops of the
   dihedral group (identity, three 90-degree rotations, two flips, two
   transpositions). These are exact index permutations with zero interpolation.
   They are geophysically valid because slope magnitude is rotation-invariant and
   bathymetry/backscatter have no preferred direction at the 128 m tile scale.

## Split rules (hard requirements for the training project)

1. **Spatial splits only.** Random tile-level splits are invalid: tiles overlap
   by 50% and augmentation passes overlap their parents, so a random split puts
   near-duplicate pixels in both train and test. With the current four polygons,
   use leave-one-polygon-out cross-validation.
2. **Augmented tiles inherit the split of their source area.** A `_rotaug` tile
   whose footprint lies in a test region is a test-region tile -- and test/val
   regions are **never** augmented at all. Only the training split may use
   `_rotaug` tiles or training-time D4 ops.
3. Every rotated/augmented manifest row carries `center_x` / `center_y` (tile
   center, UTM EPSG:32636) precisely so splits can be assigned by location.
   The `aug_pass` column identifies which pass produced a tile.

## Honest accounting

Augmentation multiplies *views*, not *information*. Roughly: ~250 base rotated
tiles -> ~1,000+ physical tiles after 4 passes -> x8 D4 ops at training time =
~10,000 effective views per epoch. These remain correlated views of ~1-2 km^2 of
labeled seabed (the reference paper trained on two orders of magnitude more
area). This is sufficient for a prototype and a baseline Dice score, not for a
generalizing production classifier. The highest-value path to a better model is
new annotated survey areas, not more augmentation.

## Storage policy

Version inputs and recipes; regenerate outputs.

- Raw `DataBase/` bundles: tracked in git LFS (canonical, small, rarely change).
- Generated `outputs/` (tiles, augmented tiles, JPEGs, stitches): **not in git**.
  Augmentation passes are deterministic, so any contributor reproduces identical
  outputs with one command:

  ```powershell
  $env:PYTHONPATH="tiling\src"; .venv\Scripts\python -m seabed_tiler --config tiling\config\polygon1.yaml --rotated --augment
  ```

- Frozen dataset releases for training experiments: zip the run directory
  (tiles + manifest), name it `<polygon>_<run_tag>_<git-commit>`, and store it on
  institutional storage / cloud -- not in git.
