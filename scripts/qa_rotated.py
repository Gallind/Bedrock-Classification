"""QA script: axis-aligned bbox vs rotated MBR overlay, plus tile-level class comparison.

Usage:
    $env:PYTHONPATH = "tiling\\src"
    .venv\\Scripts\\python scripts/qa_rotated.py --config tiling/config/polygon1.yaml

Outputs in outputs/<name>/qa_rotation/:
    bbox_vs_mbr.png          overlay: original bbox (red) vs annotation MBR (blue)
    class_distribution.txt   original vs rotated tile-level class pixel fractions
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box as shapely_box

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(Path(__file__).stem)

sys.path.insert(0, str(Path(__file__).parent.parent / "tiling" / "src"))
from seabed_tiler.logging_utils import setup_logging
from seabed_tiler.config import load_config
from seabed_tiler.align import build_grid_and_features
from seabed_tiler.rotation import compute_label_footprint, minimum_bounding_rect
from seabed_tiler.rotated_tiler import _collect_label_shapefiles, _rotated_out_dir


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser(description="QA: axis-aligned vs rotation-aware tiling.")
    ap.add_argument("--config", required=True, help="Polygon config YAML.")
    ap.add_argument("--base-dir", default=None)
    args = ap.parse_args()

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)

    orig_dir = cfg.out_dir
    rot_dir = _rotated_out_dir(cfg)
    qa_dir = base / cfg.output.dir / cfg.name / "qa_rotation"
    qa_dir.mkdir(parents=True, exist_ok=True)

    if not orig_dir.exists():
        logger.error("original tiles not found: %s -- run without --rotated first", orig_dir)
        sys.exit(1)
    if not rot_dir.exists():
        logger.error("rotated tiles not found: %s -- run with --rotated first", rot_dir)
        sys.exit(1)

    logger.info("computing annotation footprint and MBR...")
    shapefiles = _collect_label_shapefiles(cfg)
    footprint = compute_label_footprint(shapefiles)
    mbr, theta, _ = minimum_bounding_rect(footprint)

    grid = build_grid_and_features(cfg)
    feat_stack = np.stack(list(grid["features"].values()), axis=0)
    xmin, ymin, xmax, ymax = grid["extent"]
    orig_bbox = shapely_box(xmin, ymin, xmax, ymax)

    fig, ax = plt.subplots(figsize=(10, 10))
    valid_mask = np.all(feat_stack != grid["nodata"], axis=0)
    ax.imshow(
        valid_mask.astype("uint8"),
        extent=[xmin, xmax, ymin, ymax],
        origin="lower", cmap="Greys", alpha=0.4,
    )
    bx, by = orig_bbox.exterior.xy
    ax.plot(bx, by, "r-", linewidth=2, label="axis-aligned bbox")
    mx, my = mbr.exterior.xy
    ax.plot(mx, my, "b-", linewidth=2, label=f"annotation MBR (theta={np.degrees(theta):.1f} deg)")
    fx, fy = footprint.exterior.xy
    ax.fill(fx, fy, alpha=0.15, color="green", label="annotation footprint")
    ax.legend()
    ax.set_title(f"{cfg.name}: axis-aligned bbox vs annotation MBR")
    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")
    overlay_path = qa_dir / "bbox_vs_mbr.png"
    fig.savefig(overlay_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved overlay -> %s", overlay_path)

    orig_csv = orig_dir / "manifest.csv"
    rot_csv = rot_dir / "manifest.csv"

    if not orig_csv.exists() or not rot_csv.exists():
        logger.warning("manifest.csv not found -- skipping class distribution comparison")
        return

    orig_df = pd.read_csv(orig_csv)
    rot_df = pd.read_csv(rot_csv)

    class_cols = [c for c in orig_df.columns if c.endswith("_px")]
    lines = [f"Class distribution comparison: {cfg.name}\n\n"]
    lines.append(f"  Axis-aligned: {len(orig_df)} tiles\n")
    lines.append(f"  Rotated:      {len(rot_df)} tiles\n\n")

    orig_total = orig_df[class_cols].sum().sum()
    rot_class_cols = [c for c in class_cols if c in rot_df.columns]
    rot_total = rot_df[rot_class_cols].sum().sum() if rot_class_cols else 0

    for col in class_cols:
        orig_frac = orig_df[col].sum() / orig_total if orig_total > 0 else float("nan")
        if col in rot_df.columns and rot_total > 0:
            rot_frac = rot_df[col].sum() / rot_total
        else:
            rot_frac = float("nan")
        lines.append(f"  {col:<20s}  orig={orig_frac:.3f}  rot={rot_frac:.3f}\n")

    result = "".join(lines)
    logger.info(result)
    (qa_dir / "class_distribution.txt").write_text(result, encoding="utf-8")
    logger.info("QA complete -> %s", qa_dir)


if __name__ == "__main__":
    main()
