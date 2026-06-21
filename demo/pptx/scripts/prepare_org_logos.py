"""Normalize raw partner logos into deck-ready PNGs.

Drop whatever you have into demo/brand/logos/ (any of .png/.svg/.jpg/.jpeg,
any colour, any background) named iolr / reichman / code4good. This script
turns each into a transparent PNG styled for the dark slides and writes it to
demo/brand/logos/ready/<name>.png (which the deck references).

    .venv\\Scripts\\python scripts/prepare_org_logos.py

Per-logo treatment (see LOGOS below):
  "whiten"    — recolour the mark to off-white (for single-colour wordmarks
                that would vanish on a dark slide). Any flat background is
                keyed out first.
  "flood_key" — key out only the border-connected background colour, keeping
                the original colours and any interior same-colour detail
                (e.g. a white icon inside a coloured disc).
"""
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LOGOS_DIR = Path(__file__).resolve().parents[1].parent / "brand" / "logos"
READY_DIR = LOGOS_DIR / "ready"
WHITE = (238, 245, 247)          # palette text colour
EXTS = (".png", ".svg", ".jpg", ".jpeg")

LOGOS = {
    "iolr": "whiten",
    "reichman": "whiten",
    "code4good": "flood_key",
}


def _find_raw(stem: str) -> Path | None:
    for p in LOGOS_DIR.iterdir():
        if p.is_file() and p.stem.lower() == stem and p.suffix.lower() in EXTS:
            return p
    return None


def _load_rgba(path: Path) -> Image.Image:
    if path.suffix.lower() == ".svg":
        import resvg_py
        svg = path.read_text(encoding="utf-8")
        svg = re.sub(r'width="[0-9.]+mm"', 'width="1200"', svg, count=1)
        svg = re.sub(r'height="[0-9.]+mm"', 'height="1200"', svg, count=1)
        png = bytes(resvg_py.svg_to_bytes(svg_string=svg, width=1200))
        tmp = READY_DIR / f"_{path.stem}_raster.png"
        READY_DIR.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(png)
        im = Image.open(tmp).convert("RGBA")
        tmp.unlink(missing_ok=True)
        return im
    return Image.open(path).convert("RGBA")


def _flood_key(im: Image.Image, thresh: int = 40) -> Image.Image:
    """Make the border-connected background colour transparent."""
    rgb = im.convert("RGB")
    sentinel = (1, 254, 2)
    w, h = rgb.size
    for xy in ((0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)):
        ImageDraw.floodfill(rgb, xy, sentinel, thresh=thresh)
    keyed = np.array(im)
    keyed[np.all(np.array(rgb) == sentinel, axis=-1), 3] = 0
    return Image.fromarray(keyed, "RGBA")


def _whiten(im: Image.Image) -> Image.Image:
    arr = np.array(im)
    corner = arr[0, 0]
    if corner[3] > 200:                       # opaque background -> key it first
        im = _flood_key(im)
        arr = np.array(im)
    opaque = arr[:, :, 3] > 0
    arr[opaque, 0], arr[opaque, 1], arr[opaque, 2] = WHITE
    return Image.fromarray(arr, "RGBA")


def main() -> int:
    READY_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    for stem, mode in LOGOS.items():
        raw = _find_raw(stem)
        if raw is None:
            missing.append(stem)
            continue
        im = _load_rgba(raw)
        out = _whiten(im) if mode == "whiten" else _flood_key(im)
        dst = READY_DIR / f"{stem}.png"
        out.save(dst)
        print(f"  + {dst.name:16s} <- {raw.name}  ({mode})")
    if missing:
        print(f"  ! missing raw logo(s): {', '.join(missing)} "
              f"(drop a .png/.svg/.jpg into {LOGOS_DIR})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
