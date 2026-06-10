# Recreating the Training Dataset

Audience: team members who want to train a seabed classification model on this
data. This guide takes you from an empty machine to a verified, byte-identical
copy of the training dataset. Setup is about 15 minutes plus processing time.

Everything you need is in the git repository -- code, configs, and the complete
raw survey data (`DataBase/`, with `.xyz` point grids and `.jpg` renders stored
via Git LFS). The generated tiles are deliberately NOT in git: they are
deterministic, so everyone regenerates identical outputs locally with one command.

## 1. Prerequisites

- Git and [Git LFS](https://git-lfs.github.com) (`winget install GitHub.GitLFS`)
- Python 3.12 or 3.13 (rasterio/geopandas binary wheels must be available for
  your version -- do not build GDAL from source)

## 2. Clone and pull the data

```powershell
git clone <repo-url>
cd Bedrock-Classification
git lfs install
git lfs pull
```

Without `git lfs pull` the `.xyz`/`.jpg` files are small text pointer stubs and
the tiler will fail to read them.

## 3. Python environment

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r tiling\requirements.txt
```

## 4. Generate the dataset

```powershell
$env:PYTHONPATH="tiling\src"
foreach ($p in 1,3,4,5) {
    .venv\Scripts\python -m seabed_tiler --config "tiling\config\polygon$p.yaml" --rotated --augment
}
```

Each polygon produces three run directories under `outputs/<polygon>/`:

| Directory | Contents | Use for |
|---|---|---|
| `t128m_o50pct_r1m` | axis-aligned grid | QA / baseline comparison |
| `t128m_o50pct_r1m_rot` | grid aligned to the annotation footprint | base train/val/test tiles |
| `t128m_o50pct_r1m_rotaug` | deterministic augmentation passes | training split ONLY |

**Already generated outputs before? No manual cleanup is needed.** Every run
deletes its own run directory before writing, so re-running on a newer commit
fully replaces stale tiles. (Earlier versions of the tiler only overwrote files
they rewrote, which left orphan tiles from previous runs on disk -- that bug is
fixed in code. If you generated outputs on a commit older than this guide,
simply re-run the command above.) Run directories for other tile geometries
(different `run_tag`) are kept intentionally for side-by-side comparison.

JPEG previews of every tile are written to `<run dir>/jpg/` for visual inspection.

## 5. Verify your dataset

Run the test suite:

```powershell
.venv\Scripts\python -m pytest tiling\tests -q
```

Then check tile counts (rows in each `manifest.csv`, header excluded):

| Polygon | standard | `_rot` | `_rotaug` |
|---|---|---|---|
| polygon1 | 190 | 154 | 574 |
| polygon3 | 65 | 58 | 227 |
| polygon4 | 29 | 26 | 59 |
| polygon5 | 31 | 12 | 46 |

If your counts differ, you are on a different commit or `git lfs pull` did not
complete -- the pipeline is deterministic, so identical inputs give identical
outputs.

## 6. What a training sample looks like

Each tile is a co-registered pair, 128 x 128 px at 1 m/px, EPSG:32636:

- `tiles/features/<tile_id>.tif` -- multiband float32. Band order and names are
  stored in the GeoTIFF band descriptions (typically backscatter, bathymetry,
  slope). Values are raw physical units (dB, meters, degrees) -- normalize in
  your training code. Nodata = `-9999.0`.
- `tiles/labels/<tile_id>.tif` -- single band uint8: 0=background, 1=rock,
  2=shallow_rock, 3=sand. Note that 0 is BOTH the background class and the
  nodata value (see rule 3 below).
- `manifest.csv` -- one row per tile: `valid_frac`, per-class pixel counts,
  tile center in UTM (`center_x`/`center_y`), and for augmented tiles the
  `aug_pass` provenance.

## 7. Hard rules for training (binding contract)

These come from `docs/DATA_AUGMENTATION.md` -- read it in full before building
a data loader or split. Violating them silently invalidates your results:

1. **Spatial splits only.** Tiles overlap 50%, so random tile-level splits leak
   near-duplicate pixels between train and test. With four polygons, use
   leave-one-polygon-out cross-validation (assign by `center_x`/`center_y`).
2. **Augmentation is train-only.** `_rotaug` tiles and training-time D4 ops may
   only appear in the training split. Val/test use base `_rot` tiles from
   held-out regions.
3. **Mask the loss by feature validity.** Wherever any feature band is nodata
   (`-9999.0` or NaN), the label pixel must not contribute to the loss --
   otherwise the model learns that "no data" means "background seabed".
4. **Only rigid geometric augmentation.** At training time use the 8 exact D4
   ops from `seabed_tiler.augment` (numpy-only, importable without geospatial
   dependencies):

   ```python
   from seabed_tiler.augment import augment_pair, random_d4
   feats_aug, label_aug = augment_pair(features, label, random_d4(rng))
   ```

   Photometric transforms (brightness, noise, depth offsets, zoom) are
   prohibited -- pixel values are physical measurements.

## 8. Sharing a frozen dataset

For experiments where everyone must use byte-identical tiles, zip the run
directory, name it with the git commit, and attach it to a GitHub Release --
do not commit generated outputs to the repository. See "Storage policy" in
`docs/DATA_AUGMENTATION.md`.
