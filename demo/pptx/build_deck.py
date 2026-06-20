"""CLI entry point for the seabed-classification slide deck.

    python build_deck.py                 # -> out/seabed-deck.pptx
    python build_deck.py -o talk.pptx    # custom output path

Run from demo/pptx/ (so the `deck` package is importable). Sources its content,
palette, assets, and narration from demo/shared/ — the same layer the Remotion
video uses. Missing source images render as labelled placeholders rather than
failing the build.
"""
import argparse

from deck.assets import ASSETS, resolve
from deck.build import build


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the seabed-classification .pptx deck.")
    ap.add_argument("-o", "--out", default=None,
                    help="output .pptx path (default: out/seabed-deck.pptx)")
    args = ap.parse_args()

    missing = [name for name in ASSETS if resolve(name) is None]
    out, n = build(args.out)

    print(f"[+] wrote {n}-slide deck -> {out}")
    if missing:
        print(f"[!] {len(missing)} asset(s) missing — rendered as placeholders:")
        for name in missing:
            print(f"    - {name}  ({ASSETS[name]})")
        print("    Hint: run `git lfs pull`, regenerate reports/, or run the demo's copy-assets.")


if __name__ == "__main__":
    main()
