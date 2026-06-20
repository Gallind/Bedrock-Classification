"""Background keying for photographic / map images.

Classified maps sit on black, band renders on white. For those we key the
flat background colour to transparent so the subject floats on the slide
gradient instead of sitting in a hard rectangle. Charts (matplotlib output)
are graphical content and are NOT keyed — they belong on a light card.

Processed PNGs are cached under out/_assets/ (gitignored).
"""
from pathlib import Path

import numpy as np
from PIL import Image

from .paths import OUT_DIR

_CACHE = OUT_DIR / "_assets"


def keyed_transparent(src: Path, thresh: int = 42) -> Path:
    """Return a cached RGBA copy of `src` with its flat corner colour keyed out.

    The background colour is sampled from the top-left corner; every pixel within
    `thresh` (per-channel, summed) of it becomes fully transparent. Works for
    both black-backed maps and white-backed renders.
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    dst = _CACHE / f"{src.stem}_keyed.png"
    if dst.exists() and dst.stat().st_mtime >= src.stat().st_mtime:
        return dst

    im = Image.open(src).convert("RGBA")
    arr = np.asarray(im).astype(np.int16)
    bg = arr[0, 0, :3]
    dist = np.abs(arr[:, :, :3] - bg).sum(axis=2)
    out = arr.copy()
    out[dist <= thresh * 3, 3] = 0
    Image.fromarray(out.astype("uint8"), "RGBA").save(dst)
    return dst
