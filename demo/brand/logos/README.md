# Partner / organisation logos

Drop the three org logos here named `iolr`, `reichman`, `code4good`. **Any
format and any colour is fine** — `.png`, `.svg`, `.jpg`, coloured, white
background, whatever you can source. A prep script normalizes them for the dark
slides.

## Normalize them

```powershell
# from demo/pptx/
.venv\Scripts\python scripts/prepare_org_logos.py
```

This writes deck-ready transparent PNGs into `ready/` (which the deck actually
uses, via `demo/shared/assets.json`):

| Logo | Treatment |
| --- | --- |
| `iolr`, `reichman` | recoloured to off-white (single-colour wordmarks that would vanish on a dark slide) |
| `code4good` | only the outer background keyed out — keeps the teal disc and the white hand inside it |

Re-run it any time you replace a raw file. To change how a logo is treated
(e.g. keep one in colour), edit the `LOGOS` table at the top of the script.

## Where they appear

A centered strip near the bottom of the **title** and **thank-you** slides. Any
logo whose raw file is missing is simply skipped — the deck still builds.

Raw sources currently here: `iolr.png` (navy), `reichman.svg` (blue),
`Code4Good.jpg` (white background). `ready/` is regenerated, not hand-edited.
