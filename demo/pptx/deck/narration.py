"""Narration script, shared with the video's voiceover.

demo/shared/narration.json is the single source for the spoken script. The deck
reuses each scene's narration as the slide's speaker notes (via `note()`), so a
presenter reads the same words the video narrates.
"""
import json

from .paths import SHARED_DIR

_scenes = json.loads((SHARED_DIR / "narration.json").read_text(encoding="utf-8"))

BY_ID: dict[str, dict] = {s["id"]: s for s in _scenes}


def note(scene_id: str | None) -> str:
    """Narration text for a scene id, or '' if unknown / not provided."""
    if not scene_id:
        return ""
    scene = BY_ID.get(scene_id)
    return scene["narration"] if scene else ""
