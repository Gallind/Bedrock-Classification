# Re-training the U-Net on an RTX 4050 (CUDA)

Audience: whoever runs the U-Net retraining on the GPU box. The RF/HGB tree
baseline is CPU-only and runs on the Mac (`docs/` has the forest sweep in the
combined retrain plan); this guide covers **only** the U-Net half on a CUDA GPU.

## Why retrain

The polygon1 label fix (commit `a4aeeb5`: priority burn order + recovering the
mislabeled rock feature) changed how labels are rasterized. Every checkpoint
currently in `training/runs/` was fit on the *old* labels, so the U-Net must be
retrained on the corrected polygon1 tiles. This is a full refresh: both the
3-band and 2-band experiments, each with train + eval + polygon4 prediction +
leave-one-polygon-out (LOPO) folds + summaries.

Determinism: `train` wipes its run dir on every invocation (no resume) and seeds
everything with `seed: 42`, so reruns never mix artifacts.

## 1. Prerequisite — tiles must exist on the box

The U-Net consumes the tiler outputs `outputs/<polygon>/t128m_o50pct_r1m_rot/`
(`_rot`, the train/val/test base) and `_rotaug/` (`_rotaug`, train-only D4
augmentation) for polygons 1/3/4/5. Tiles are deterministic and gitignored, so
you get identical bytes whether you regenerate or copy. Pick one:

**Option A — regenerate on the box** (needs the tiler venv `.venv`, Python 3.12/3.13,
`PYTHONPATH=tiling/src`):

```bash
export PYTHONPATH=tiling/src
for p in polygon1 polygon3 polygon4 polygon5; do
  .venv/bin/python -m seabed_tiler --config tiling/config/$p.yaml --rotated   # _rot
  .venv/bin/python -m seabed_tiler --config tiling/config/$p.yaml --augment   # _rotaug
done
```

**Option B — copy** the regenerated `outputs/` tree from the Mac (rsync/scp). The
tiler is deterministic, so both options give byte-identical tiles.

The U-Net does **not** need polygon6 (it trains on 1/3/4/5 only; polygon6 is
forest-only).

## 2. Environment on the RTX box (one-time)

The repo `.venv-train` is macOS-specific (CPU/MPS torch) — do not reuse it. Create
a fresh Python 3.12 venv with a **CUDA** torch build:

```bash
python3.12 -m venv .venv-train
.venv-train/bin/pip install -r training/requirements.txt   # pins torch==2.2.2, numpy<2
```

- **Linux:** the default `torch==2.2.2` wheel is already `cu121` (CUDA) → works on
  the RTX 4050 (sm_89) out of the box.
- **Windows:** the default wheel is CPU-only → reinstall the CUDA build explicitly:
  ```bat
  .venv-train\Scripts\pip install --force-reinstall torch==2.2.2 ^
      --index-url https://download.pytorch.org/whl/cu121
  ```

Verify the GPU is visible (must print `True` and the card name):

```bash
.venv-train/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

The model is tiny (`base_filters: 16`, `depth: 4`, ~0.5M params, batch 16 @ 128px),
so it fits the RTX 4050's VRAM comfortably — no batch-size reduction needed.

## 3. Run the full sweep

```bash
export PYTHONPATH=tiling/src:training/src
.venv-train/bin/python -m seabed_unet.run_all \
  --configs training/config/experiment_3band.yaml training/config/experiment_2band.yaml \
  --lanes cuda,cpu,cpu
```

**The `--lanes cuda,cpu,cpu` override is mandatory.** `run_all.py` defaults
`--lanes` to `mps,cpu,cpu` (set for the Mac) and appends `--device <lane>` to each
job. Without the override the GPU lane launches `--device mps` and fails on the
RTX box. The `cuda` lane drives the GPU for the long training jobs while the two
`cpu` lanes fill with LOPO/eval jobs (disjoint OMP/MKL budgets) so the GPU never
idles.

## 4. Outputs

Per experiment (`unet_3band`, `unet_2band`) under `training/runs/<name>/`:

- `best.pt` — checkpoint at best val macro-Dice
- `normalization_stats.json` — per-band stats for inference
- `history.csv` — per-epoch loss / val-Dice / lr curves
- `train.log` — should show `device=cuda`
- `eval_test/metrics.json` + confusion matrix
- `maps/` — polygon4 prediction GeoTIFF + JPEG
- LOPO `summary_*.json` (per-fold macro-Dice); runner logs under `_runner/`

`training/runs/` is gitignored and never committed — share frozen checkpoints via
external storage, not git.

## 5. Verification

```bash
PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m pytest training/tests -q
```

Then confirm, for each of `training/runs/unet_3band/` and `unet_2band/`:

- a fresh `best.pt` exists;
- `history.csv` is non-trivial (loss decreasing, early stop fired);
- `eval_test/metrics.json` is present, and LOPO `summary_*.json` macro-Dice looks
  reasonable;
- `train.log` shows `device=cuda` (proves the GPU lane actually ran on CUDA).
- **Sanity vs the labels fix:** polygon1 metrics (rock recall in particular) should
  move relative to the previous round, reflecting the corrected labels.
