# Web demo — "Watch the seabed classify"

A static website that replays each model classifying a polygon **tile by tile**, like the
desktop `seabed_unet.watch` / `seabed_forest.watch` viewers — but with **no inference at
request time**. The heavy work (torch + sklearn + rasterio) runs once, offline, in a recorder
that writes compact static assets + JSON. The site just plays them back, so it needs zero ML
dependencies to host.

```
webapp/
  data/            # GENERATED recordings (gitignored) — produced by the recorder
    catalog.json   #   index of polygons + models
    sessions/<polygon>/
      manifest.json    #   playback timeline for that polygon
      backdrop.jpg     #   grayscale survey mosaic (shared)
      truth.png        #   ground-truth overlay (RGBA, transparent where unlabeled)
      steps/...        #   per-tile band thumbnails + per-model cumulative maps & tile preds
  web/             # frontend (built separately via Claude Design — see prompt in the plan)
  README.md
```

## 1. Record (offline, in the training venv)

From the repo root, with both src dirs on `PYTHONPATH`:

```bash
export PYTHONPATH=tiling/src:training/src
.venv-train/bin/python -m seabed_unet.export \
    --polygon polygon4 --polygon polygon3 --polygon polygon5
```

Each polygon is recorded with **all available model lanes** in lockstep (one shared tile
sequence): `unet_3band`, `unet_2band`, `rf_raw`, `rf_spatial`, `hgb_raw`, `hgb_spatial`. A lane
is skipped (with a warning) if its artifact is missing — U-Net lanes need
`training/runs/unet_{3,2}band/best.pt`; tree lanes need
`training/runs/forest_3band/model_{random_forest,hist_gradient_boosting}.joblib`. After the
requested polygons are recorded it rewrites `data/catalog.json` from every manifest on disk, so
you can record one polygon at a time and the catalog stays complete.

Useful flags:

| flag | default | purpose |
|------|---------|---------|
| `--data-dir` | `webapp/data` | output root |
| `--max-long-side` | `1400` | cap the map grid's long side (px) |
| `--models` | all | comma subset of lane ids to record |
| `--max-tiles` | all | record only the first N tiles (smoke test) |
| `--device` | auto | U-Net device (`cpu`/`mps`/`cuda`) |
| `--forest-config` / `--unet-3band-config` / `--unet-2band-config` | repo configs | source configs |

Log: `webapp/data/export.log`.

## 2. Host (no server needed)

Any static host works. For local dev:

```bash
cd webapp && python -m http.server 8000
# frontend served from web/, data fetched from /data (DATA_BASE_URL)
```

The frontend fetches `${DATA_BASE_URL}/catalog.json` then a polygon's
`sessions/<polygon>/manifest.json` and the image assets it references (paths are relative to
that session dir). All map images (`backdrop.jpg`, `truth.png`, every `class_map`) share the
manifest's `map_size` grid, so the client overlays them pixel-aligned and draws the yellow tile
outline from each step's `outline_px`.

## Notes
- `data/` is regenerated and **gitignored** (like `outputs/` and `training/runs/`). Share frozen
  recordings as a zipped `data/` named with the git commit, not via git.
- Guided-spatial lanes re-regularize the whole posterior every tile, so recording them is the
  slowest part; use `--models` to skip them for a quick pass.
- The frontend lives in `web/` and is generated with Claude Design — the paste-ready prompt is
  in the plan at `~/.claude/plans/i-want-to-create-wondrous-lampson.md` (the data contract there
  matches `catalog.json` / `manifest.json` exactly).
