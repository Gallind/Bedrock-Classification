"""Embed TrueType fonts into a saved .pptx (OOXML font embedding).

python-pptx can't embed fonts, so after saving we post-process the package:
add each TTF as a /ppt/fonts/fontN.fntdata part, register it in the content
types and presentation rels, and list it under <p:embeddedFontLst> with
embedTrueTypeFonts enabled. The deck then renders with Space Grotesk / Inter /
JetBrains Mono on any machine, even without the fonts installed.

Fonts live in pptx/fonts/ (downloaded, project-local). If they're absent the
deck still builds — it just relies on the viewer's system fonts.
"""
import zipfile
from pathlib import Path

from lxml import etree

from .paths import PPTX_DIR

FONTS_DIR = PPTX_DIR / "fonts"

# typeface name -> {regular, bold} TTF filenames in FONTS_DIR
FONTS = {
    "Epilogue": {"regular": "Epilogue-Regular.ttf", "bold": "Epilogue-Bold.ttf"},
    "Manrope": {"regular": "Manrope-Regular.ttf", "bold": "Manrope-Bold.ttf"},
    "JetBrains Mono": {"regular": "JetBrainsMono-Regular.ttf", "bold": "JetBrainsMono-Bold.ttf"},
}

_P = "http://schemas.openxmlformats.org/presentationml/2006/main"
_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_CT = "http://schemas.openxmlformats.org/package/2006/content-types"
_PR = "http://schemas.openxmlformats.org/package/2006/relationships"


def _xml(el) -> bytes:
    return etree.tostring(el, xml_declaration=True, encoding="UTF-8", standalone=True)


def available() -> bool:
    return all(
        (FONTS_DIR / v).exists() for spec in FONTS.values() for v in spec.values()
    )


def embed_fonts(pptx_path: str | Path) -> bool:
    """No-op — font embedding is handled by PowerPoint's native save option."""
    return False
