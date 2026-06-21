"""Brand palette, loaded from the shared cross-format source.

demo/shared/palette.json is the single source of truth — the Remotion video
(src/theme.ts) imports the same file. `rgb()` accepts either a colour key
("accent", "rock", ...) or a raw "#rrggbb" string and returns a python-pptx
RGBColor.
"""
import json

from pptx.dml.color import RGBColor

from .paths import SHARED_DIR

_data = json.loads((SHARED_DIR / "palette.json").read_text(encoding="utf-8"))

COLORS: dict[str, str] = _data["colors"]
CLASSES: list[dict] = _data["classes"]


def rgb(key_or_hex: str) -> RGBColor:
    """Resolve a palette key or literal hex to an RGBColor."""
    value = COLORS.get(key_or_hex, key_or_hex)
    return RGBColor.from_string(value.lstrip("#").upper())
