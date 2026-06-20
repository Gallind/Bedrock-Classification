# Seabed Classification — Slide Deck (`demo/pptx/`)

A [python-pptx](https://python-pptx.readthedocs.io) generator that builds a
PowerPoint deck for the seabed-classification project. It is a **sibling** of the
Remotion video (`demo/src/`), not a fork of it: both read the shared content
layer in [`demo/shared/`](../shared), but the deck lays out its own slides — a
different order with a bit more detail than the ~5-minute video.

```
demo/
  shared/            cross-format source, read by BOTH renderers
    narration.json   the spoken script (also the video's TTS source)
    palette.json     brand colours + class colours (also src/theme.ts)
    assets.json      logical name -> repo path (also scripts/copy-assets.mjs)
  src/               Remotion video (React -> MP4)
  pptx/              this deck generator (Python -> PPTX)
    deck/
      paths.py       repo/shared path anchors
      palette.py     loads shared/palette.json
      assets.py      loads shared/assets.json, resolves repo images
      narration.py   loads shared/narration.json (-> speaker notes)
      content.py     THE DECK: ordered, typed slide specs (edit here)
      render.py      draws each slide spec with python-pptx
      build.py       assembles + saves the .pptx
    build_deck.py    CLI entry point
  brand/             logo SVG/PNG assets (referenced via shared/assets.json)
```

## What is shared vs separate

| Resource | Shared with the video | Deck-specific |
| --- | --- | --- |
| Narration / speaker notes | `shared/narration.json` | — |
| Palette + class colours | `shared/palette.json` | — |
| Image assets + logical names | `shared/assets.json` | — |
| Slide order & wording | — | `deck/content.py` |
| Layout & styling | — | `deck/render.py` |

Edit content in `content.py`; edit shared text/colours/asset paths in
`demo/shared/` and the video picks up the same change.

## Setup

```powershell
# from demo/pptx/
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```

## Build

```powershell
# from demo/pptx/
.venv\Scripts\python build_deck.py            # -> out/seabed-deck.pptx
.venv\Scripts\python build_deck.py -o talk.pptx
```

The source images are the repo's real files (LFS): run `git lfs pull` at the
repo root first if you only see pointer text. Any missing image renders as a
labelled placeholder instead of failing the build — the CLI lists what was
missing at the end.

`out/` and `.venv/` are gitignored — the `.pptx` is a regenerated artifact, not
committed (same policy as the video's `out/` and the pipeline's `outputs/`).

## Metrics note

The numbers in `content.py` are the originally published results (within-survey
macro-Dice 0.784, cross-survey LOPO 0.608), matching the already-distributed
video. They are written in, not auto-derived from `training/runs/` — update
`content.py` when refreshing the deck.
