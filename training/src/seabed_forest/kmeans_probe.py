"""K-means unsupervised probe for seabed classification.

Clusters all labeled pixels by their normalized feature values (backscatter,
bathymetry, slope), then matches clusters to ground-truth classes via the
Hungarian algorithm (optimal bijection). Reports per-class Dice and produces
a side-by-side comparison plot (ground truth vs K-means clusters) per polygon.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/Scripts/python -m seabed_forest.kmeans_probe \
      --config training/config/forest_3band.yaml [--k 3] [--out training/runs/kmeans]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from seabed_unet.config import Config
from seabed_unet.data import features_by_polygon, load_split_records
from seabed_unet.logging_utils import setup_logging
from seabed_unet.metrics import confusion_matrix, metrics_report
from seabed_unet.normalize import compute_stats

from .config import load_forest_config
from .pixels import build_pixel_table

logger = logging.getLogger(__spec__.name if __spec__ is not None else __name__)

CLASS_COLORS = {
    0: (68, 114, 196),    # rock — blue
    1: (237, 125, 49),    # shallow_rock — orange
    2: (112, 173, 71),    # sand — green
    -1: (30, 30, 30),     # unmatched
}
CLUSTER_COLORS = [
    (220, 50, 50),
    (50, 180, 220),
    (220, 200, 50),
    (180, 50, 220),
    (50, 220, 120),
    (220, 120, 50),
]


def _hungarian_match(cm: np.ndarray) -> dict[int, int]:
    """Match cluster indices to class indices maximising total overlap (greedy approx)."""
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-cm)
    return {int(r): int(c) for r, c in zip(row_ind, col_ind)}


def _colorize(labels: np.ndarray, color_map: dict[int, tuple]) -> np.ndarray:
    h, w = labels.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in color_map.items():
        mask = labels == val
        rgb[mask] = color
    return rgb


def _render_polygon_comparison(
    polygon: str,
    coords: np.ndarray,
    y_true: np.ndarray,
    cluster_mapped: np.ndarray,
    class_names: list[str],
    out_dir: Path,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    # Build pixel grids by snapping to 1 m grid
    xs = coords[:, 0]
    ys = coords[:, 1]
    x0, y0 = xs.min(), ys.min()
    cols = ((xs - x0)).astype(int)
    rows = ((ys - y0)).astype(int)
    H = rows.max() + 1
    W = cols.max() + 1

    gt_grid = np.full((H, W), -1, dtype=np.int64)
    km_grid = np.full((H, W), -1, dtype=np.int64)
    gt_grid[rows, cols] = y_true
    km_grid[rows, cols] = cluster_mapped

    gt_colors = {i: CLASS_COLORS[i] for i in range(len(class_names))}
    gt_colors[-1] = (20, 20, 20)
    km_colors = {i: CLASS_COLORS[i] for i in range(len(class_names))}
    km_colors[-1] = (20, 20, 20)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(_colorize(gt_grid, gt_colors), origin="lower", aspect="equal")
    axes[0].set_title(f"{polygon} — ground truth")
    axes[0].axis("off")

    axes[1].imshow(_colorize(km_grid, km_colors), origin="lower", aspect="equal")
    axes[1].set_title(f"{polygon} — K-means (matched)")
    axes[1].axis("off")

    patches = [mpatches.Patch(color=[c/255 for c in CLASS_COLORS[i]], label=class_names[i])
               for i in range(len(class_names))]
    fig.legend(handles=patches, loc="lower center", ncol=len(class_names), frameon=False)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = out_dir / f"{polygon}_kmeans_vs_gt.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"    plot -> {out_path}")


def run_kmeans_probe(cfg: Config, k: int, out_dir: Path) -> None:
    from sklearn.cluster import KMeans

    out_dir.mkdir(parents=True, exist_ok=True)
    band_modes = cfg.normalization.modes_for(cfg.bands)
    splits = load_split_records(cfg)
    all_records = [r for split in splits.values() for r in split]

    train_features = [r.features for r in all_records if not r.augmented]
    stats = compute_stats(
        features_by_polygon(splits), train_features, cfg.bands, band_modes,
        cfg.feature_nodata, tuple(cfg.normalization.clip_percentiles),
    )

    # Build full pixel table (all splits, deduped)
    X, y, coords = build_pixel_table(
        [r for r in all_records if not r.augmented],
        cfg.bands, cfg.class_ids, stats, cfg.feature_nodata,
        cfg.ignore_label, band_modes, dedup=True, seed=42,
    )
    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]

    logger.info(f"[+] K-means probe: {X.shape[0]:,} pixels, K={k}, bands={cfg.bands}")
    logger.info(f"    class counts: { {class_names[c]: int(n) for c, n in zip(*np.unique(y, return_counts=True))} }")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = km.fit_predict(X)
    logger.info(f"    K-means inertia: {km.inertia_:.1f}")

    # Build overlap matrix: (k, num_classes) — rows=clusters, cols=true classes
    overlap = np.zeros((k, cfg.num_classes), dtype=np.int64)
    for cl, gt in zip(cluster_labels, y):
        overlap[cl, gt] += 1

    match = _hungarian_match(overlap)
    logger.info(f"    cluster->class assignment: { {f'cluster{c}': class_names[cls] for c, cls in match.items()} }")

    # Map cluster labels to matched class indices
    mapped = np.array([match.get(int(cl), -1) for cl in cluster_labels], dtype=np.int64)

    # Overall metrics
    valid = mapped >= 0
    cm = confusion_matrix(mapped[valid], y[valid], cfg.num_classes)
    report = metrics_report(cm, class_names)
    logger.info(f"    macro-Dice (matched): {report['macro_dice']:.4f}  OA: {report['overall_accuracy']:.4f}")
    for cls in class_names:
        logger.info(f"      {cls}: dice {report['per_class'][cls]['dice']:.4f}")

    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))

    # Per-polygon plots
    polygon_ids = np.array([r.polygon for r in [r for r in all_records if not r.augmented]
                            for _ in range(0)])  # placeholder

    # Rebuild per-polygon for plotting
    _render_per_polygon(
        [r for r in all_records if not r.augmented],
        cfg, stats, band_modes, km, match, class_names, k, out_dir,
    )

    logger.info(f"[+] done -> {out_dir}")
    return report


def _render_per_polygon(records, cfg, stats, band_modes, km, match, class_names, k, out_dir):
    from seabed_unet.normalize import apply_stats
    from seabed_unet.data import encode_target, IGNORE_INDEX

    polygons = sorted({r.polygon for r in records})
    for polygon in polygons:
        poly_records = [r for r in records if r.polygon == polygon]
        xs_list, ys_list, coords_list = [], [], []
        for r in poly_records:
            norm = apply_stats(r.features, r.polygon, cfg.bands, stats, cfg.feature_nodata, band_modes)
            target = encode_target(r.label, r.features, cfg.class_ids, cfg.feature_nodata, cfg.ignore_label)
            valid = target != IGNORE_INDEX
            if not valid.any():
                continue
            rows_px, cols_px = np.nonzero(valid)
            X_px = norm[:, rows_px, cols_px].T.astype(np.float32)
            y_px = target[rows_px, cols_px]
            cc = cols_px.astype(np.float64) + 0.5
            rr = rows_px.astype(np.float64) + 0.5
            t = r.transform
            cx = t.a * cc + t.b * rr + t.c
            cy = t.d * cc + t.e * rr + t.f
            coord_px = np.column_stack([cx, cy])
            xs_list.append(X_px)
            ys_list.append(y_px)
            coords_list.append(coord_px)

        if not xs_list:
            continue
        X_poly = np.concatenate(xs_list)
        y_poly = np.concatenate(ys_list)
        coords_poly = np.concatenate(coords_list)

        # Dedup by 1m world coordinate
        keys = np.floor(coords_poly).astype(np.int64)
        _, keep = np.unique(keys, axis=0, return_index=True)
        X_poly = X_poly[keep]
        y_poly = y_poly[keep]
        coords_poly = coords_poly[keep]

        clusters = km.predict(X_poly)
        mapped = np.array([match.get(int(cl), -1) for cl in clusters], dtype=np.int64)

        valid = mapped >= 0
        if valid.sum() > 0:
            cm = confusion_matrix(mapped[valid], y_poly[valid], cfg.num_classes)
            rep = metrics_report(cm, class_names)
            logger.info(f"    {polygon}: macro-Dice {rep['macro_dice']:.4f}  "
                        + "  ".join(f"{cls}={rep['per_class'][cls]['dice']:.3f}" for cls in class_names))

        _render_polygon_comparison(polygon, coords_poly, y_poly, mapped, class_names, out_dir)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="K-means unsupervised probe for seabed data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--k", type=int, default=3, help="Number of clusters (default 3).")
    parser.add_argument("--out", default=None, help="Output directory (default: training/runs/kmeans).")
    parser.add_argument("--base-dir", default=None)
    args = parser.parse_args(argv)
    setup_logging()
    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg, _ = load_forest_config(args.config, base_dir=base)
    out_dir = Path(args.out) if args.out else base / cfg.runs_dir / "kmeans"
    run_kmeans_probe(cfg, args.k, out_dir)


if __name__ == "__main__":
    main()
