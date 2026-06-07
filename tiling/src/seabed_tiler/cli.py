"""Command-line entrypoint: align layers, rasterize labels, tile, write manifest.

Run with:  PYTHONPATH=tiling/src python -m seabed_tiler --config tiling/config/polygon1.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .align import build_grid_and_features
from .config import load_config, validate_inputs
from .labels import build_label_array
from .manifest import write_grid_preview, write_manifest, write_rotated_manifest
from .tiler import run_tiling


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Split a polygon's survey data into overlapping georeferenced tiles.")
    parser.add_argument("--config", required=True, help="Path to a polygon config YAML.")
    parser.add_argument(
        "--base-dir",
        default=None,
        help="Root that src_dir/output paths resolve against (default: cwd).",
    )
    parser.add_argument(
        "--rotated",
        action="store_true",
        help="Also produce rotation-aware tiles aligned to the annotation footprint "
             "MBR (outputs to <run_tag>_rot/). Does not replace the standard output.",
    )
    args = parser.parse_args(argv)

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)
    validate_inputs(cfg)

    print(
        f"[+] {cfg.name}: tile={cfg.tile_size_m}m stride={cfg.stride_m}m "
        f"res={cfg.target_resolution_m}m crs={cfg.crs}"
    )

    grid = build_grid_and_features(cfg)
    n_rows, n_cols = grid["shape"]
    print(f"    master grid {n_rows}x{n_cols} px, extent {grid['extent']}")
    for name, arr in grid["features"].items():
        valid = (arr != grid["nodata"]).mean()
        print(f"    layer {name:<12} coverage {valid:6.1%}")

    grid["label"] = build_label_array(cfg, grid["transform"], grid["crs"], grid["shape"])
    labeled = (grid["label"] != cfg.output.label_nodata).mean()
    print(f"    labels        coverage {labeled:6.1%}")

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    rows, windows = run_tiling(cfg, grid)

    write_manifest(rows, cfg.out_dir, grid["crs"])
    write_grid_preview(windows, cfg.out_dir, grid["crs"])

    print(f"[+] wrote {len(rows)} tiles (of {len(windows)} candidates) -> {cfg.out_dir}")

    if args.rotated:
        from .rotated_tiler import _rotated_out_dir, run_rotated_tiling
        from .stitch import stitch_rotated_features, stitch_rotated_labels
        from .viz import resolve_styles
        print("[+] rotation-aware tiling ...")
        rot_rows, _ = run_rotated_tiling(cfg, grid)
        rot_out = _rotated_out_dir(cfg)
        rot_out.mkdir(parents=True, exist_ok=True)
        res = cfg.target_resolution_m
        tpx = int(round(cfg.tile_size_m / res))
        theta_deg = rot_rows[0]["theta_deg"] if rot_rows else 0.0
        print(f"    annotation MBR theta={theta_deg:.1f} deg")
        write_rotated_manifest(rot_rows, rot_out, grid["crs"], res, tpx)
        print(f"[+] rotated: {len(rot_rows)} tiles -> {rot_out}")
        print("[+] stitching rotated tiles ...")
        styles = resolve_styles(args.config)
        stitch_out = rot_out / "stitched"
        stitch_rotated_features(rot_out, stitch_out, styles)
        stitch_rotated_labels(rot_out, stitch_out)
        print(f"[+] stitched -> {stitch_out}")
        print("[+] converting rotated tiles to JPEGs ...")
        from .to_jpg import convert_features, convert_labels
        jpg_out = rot_out / "jpg"
        convert_features(rot_out, jpg_out, limit=None, worldfile=True, styles=styles)
        convert_labels(rot_out, jpg_out, limit=None, worldfile=True)
        print(f"[+] jpg tiles -> {jpg_out}")


if __name__ == "__main__":
    main()
