"""Filesystem anchors for the deck generator.

deck/ -> pptx/ -> demo/ -> <repo root>. Everything the deck reads (shared JSON,
repo images) is resolved relative to these so the tool works from any cwd.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent          # demo/pptx/deck
PPTX_DIR = HERE.parent                           # demo/pptx
DEMO_DIR = PPTX_DIR.parent                       # demo
REPO_ROOT = DEMO_DIR.parent                      # repo root
SHARED_DIR = DEMO_DIR / "shared"                 # demo/shared (cross-format)
OUT_DIR = PPTX_DIR / "out"                        # generated decks (gitignored)
