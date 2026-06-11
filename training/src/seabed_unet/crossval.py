"""Leave-one-polygon-out cross-validation: the honest cross-survey number.

Run from repo root:
  PYTHONPATH=tiling/src:training/src .venv-train/bin/python -m seabed_unet.crossval \
      --config training/config/experiment_3band.yaml

The development split (spatial blocks) mixes every survey into training, so it
cannot measure generalization to an unseen survey. This harness retrains the
experiment 4 times with whole-polygon holdout (fold i: test = polygon_i,
val = next polygon in rotation, train = the rest) and reports mean +/- std.

Outputs: <runs_dir>/<name>_lopo/fold_<polygon>/ (one full training run each)
and <runs_dir>/<name>_lopo/summary.json + a printed markdown table.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import Config, load_config
from .evaluate import evaluate_checkpoint
from .train import run_training

FOLD_ORDER = ["polygon1", "polygon3", "polygon4", "polygon5"]


def lopo_folds(polygons: list[str]) -> list[dict[str, list[str]]]:
    """Fold i: test = polygons[i], val = next polygon (rotation), train = rest."""
    if len(polygons) < 3:
        raise ValueError(f"LOPO needs >= 3 polygons, got {polygons}")
    folds = []
    for i, test_poly in enumerate(polygons):
        val_poly = polygons[(i + 1) % len(polygons)]
        train = [p for p in polygons if p not in (test_poly, val_poly)]
        folds.append({"train": train, "val": [val_poly], "test": [test_poly]})
    return folds


def fold_config(cfg: Config, fold: dict[str, list[str]], lopo_name: str) -> Config:
    """Per-fold config: polygon-mode split, run dir nested under the LOPO root."""
    fold_cfg = cfg.model_copy(deep=True)
    fold_cfg.split = cfg.split.model_copy(
        update={
            "mode": "polygon",
            "train": fold["train"],
            "val": fold["val"],
            "test": fold["test"],
            "polygons": [],
        }
    )
    fold_cfg.name = f"{lopo_name}/fold_{fold['test'][0]}"
    return fold_cfg


def summarize(fold_reports: dict[str, dict], class_names: list[str]) -> dict:
    """mean +/- std over folds for the headline metrics."""
    def collect(path_fn):
        vals = np.array([path_fn(r) for r in fold_reports.values()], dtype=float)
        return {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals)),
                "per_fold": {k: float(path_fn(r)) for k, r in fold_reports.items()}}

    summary = {
        "overall_accuracy": collect(lambda r: r["overall_accuracy"]),
        "cohens_kappa": collect(lambda r: r["cohens_kappa"]),
        "macro_dice": collect(lambda r: r["macro_dice"]),
    }
    for name in class_names:
        summary[f"dice_{name}"] = collect(lambda r, n=name: r["per_class"][n]["dice"])
    return summary


def print_table(summary: dict) -> None:
    folds = list(next(iter(summary.values()))["per_fold"].keys())
    print("\n| metric | " + " | ".join(folds) + " | mean ± std |")
    print("|---|" + "---|" * (len(folds) + 1))
    for metric, s in summary.items():
        row = " | ".join(f"{s['per_fold'][f]:.3f}" for f in folds)
        print(f"| {metric} | {row} | {s['mean']:.3f} ± {s['std']:.3f} |")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="LOPO cross-validation for the seabed U-Net.")
    parser.add_argument("--config", required=True, help="Experiment config YAML.")
    parser.add_argument("--base-dir", default=None, help="Repo root (default: cwd).")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap tiles per split (smoke runs only — not a valid experiment).",
    )
    args = parser.parse_args(argv)

    base = Path(args.base_dir).resolve() if args.base_dir else Path.cwd()
    cfg = load_config(args.config, base_dir=base)
    if args.epochs is not None:
        cfg.train.epochs = args.epochs

    polygons = cfg.split.polygons or FOLD_ORDER
    lopo_name = f"{cfg.name}_lopo"
    folds = lopo_folds(polygons)
    print(f"[+] {lopo_name}: {len(folds)} folds over {polygons}")

    fold_reports: dict[str, dict] = {}
    for fold in folds:
        test_poly = fold["test"][0]
        print(f"\n[+] ===== fold test={test_poly} val={fold['val'][0]} "
              f"train={fold['train']} =====")
        fcfg = fold_config(cfg, fold, lopo_name)
        run_training(fcfg, limit=args.limit)
        fold_reports[test_poly] = evaluate_checkpoint(fcfg, split="test")

    class_names = [cfg.id_to_name[cid] for cid in cfg.class_ids]
    summary = summarize(fold_reports, class_names)

    lopo_dir = cfg.base_dir / cfg.runs_dir / lopo_name
    lopo_dir.mkdir(parents=True, exist_ok=True)
    (lopo_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print_table(summary)
    print(f"\n[+] LOPO summary -> {lopo_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
