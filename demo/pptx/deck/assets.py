"""Visual-asset resolution against the shared manifest.

demo/shared/assets.json maps a logical name to a repo-relative source path and
is shared with the video's copy-assets step. `resolve()` returns the absolute
path, or None if the source is missing (so a partial dataset still produces a
deck with placeholders instead of crashing).
"""
import json

from .paths import REPO_ROOT, SHARED_DIR

ASSETS: dict[str, str] = json.loads(
    (SHARED_DIR / "assets.json").read_text(encoding="utf-8")
)["assets"]


def resolve(name: str):
    """Absolute path for a logical asset name, or None if the file is missing."""
    rel = ASSETS.get(name)
    if rel is None:
        raise KeyError(
            f"unknown asset '{name}' — add it to demo/shared/assets.json"
        )
    path = REPO_ROOT / rel
    return path if path.exists() else None
