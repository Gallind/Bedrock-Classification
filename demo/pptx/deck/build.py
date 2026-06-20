"""Assemble the deck into a .pptx file."""
from pathlib import Path

from .content import DECK
from .paths import OUT_DIR
from .render import DeckBuilder


def build(out_path: str | Path | None = None, deck: list | None = None) -> tuple[Path, int]:
    """Render `deck` (defaults to content.DECK) and save a .pptx. Returns (path, n_slides)."""
    deck = deck if deck is not None else DECK
    builder = DeckBuilder()
    for spec in deck:
        builder.add(spec)
    out = Path(out_path) if out_path else OUT_DIR / "seabed-deck.pptx"
    out.parent.mkdir(parents=True, exist_ok=True)
    builder.prs.save(str(out))
    return out, len(deck)
