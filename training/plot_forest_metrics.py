"""RF vs HGB test metrics comparison.

Tree models are one-shot fits (no epoch loop / no training loss curve).
This graph shows the full per-class + aggregate test metrics side by side.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

RUNS_DIR = Path(__file__).parent / "runs" / "forest_3band"

MODELS = {
    "random_forest":        {"label": "Random Forest",             "color": "#4C9BE8"},
    "hist_gradient_boosting": {"label": "Hist Gradient Boosting", "color": "#E8834C"},
}
CLASS_NAMES = ["rock", "shallow_rock", "sand"]
CLASS_LABELS = ["Rock", "Shallow Rock", "Sand"]


def load_metrics(kind: str) -> dict:
    with open(RUNS_DIR / f"metrics_{kind}.json") as f:
        return json.load(f)


reports = {kind: load_metrics(kind) for kind in MODELS}

fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
fig.suptitle("RF vs HGB — Test Metrics (forest_3band)", fontsize=14, fontweight="bold", y=1.01)

# ── panel 1: macro-level metrics (macro dice, OA, kappa) ─────────────────────
ax = axes[0]
metrics_keys  = ["macro_dice", "overall_accuracy", "cohens_kappa"]
metric_labels = ["Macro Dice", "Overall Accuracy", "Cohen's Kappa"]
x = np.arange(len(metrics_keys))
width = 0.35

for i, (kind, meta) in enumerate(MODELS.items()):
    vals = [reports[kind][k] for k in metrics_keys]
    bars = ax.bar(x + (i - 0.5) * width, vals, width,
                  label=meta["label"], color=meta["color"], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=9)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Aggregate Metrics", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

# ── panel 2: per-class Dice ──────────────────────────────────────────────────
ax = axes[1]
x = np.arange(len(CLASS_NAMES))

for i, (kind, meta) in enumerate(MODELS.items()):
    vals = [reports[kind]["per_class"][c]["dice"] for c in CLASS_NAMES]
    bars = ax.bar(x + (i - 0.5) * width, vals, width,
                  label=meta["label"], color=meta["color"], edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(CLASS_LABELS, fontsize=9)
ax.set_ylim(0, 1)
ax.set_ylabel("Dice")
ax.set_title("Per-Class Dice", fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, axis="y", alpha=0.3)

# ── panel 3: producer's accuracy (recall) + user's accuracy (precision) ──────
ax = axes[2]
n_classes = len(CLASS_NAMES)
n_metrics = 2  # PAcc, UAcc
group_w = 0.7
bar_w = group_w / (len(MODELS) * n_metrics)
offsets = np.linspace(-group_w / 2 + bar_w / 2, group_w / 2 - bar_w / 2,
                       len(MODELS) * n_metrics)

x = np.arange(n_classes)
alphas = [1.0, 0.55]
hatches = ["", "//"]
legend_handles = []

for m_idx, (kind, meta) in enumerate(MODELS.items()):
    for p_idx, (pa_key, pa_label) in enumerate(
        [("producers_accuracy", "PAcc (recall)"), ("users_accuracy", "UAcc (precision)")]
    ):
        vals = [reports[kind]["per_class"][c][pa_key] for c in CLASS_NAMES]
        col_idx = m_idx * n_metrics + p_idx
        bars = ax.bar(
            x + offsets[col_idx], vals, bar_w,
            color=meta["color"], alpha=alphas[p_idx],
            hatch=hatches[p_idx], edgecolor="white", linewidth=0.5,
        )
        legend_handles.append(
            plt.Rectangle((0, 0), 1, 1, fc=meta["color"], alpha=alphas[p_idx],
                           hatch=hatches[p_idx], ec="grey",
                           label=f"{meta['label']} {pa_label}")
        )

ax.set_xticks(x)
ax.set_xticklabels(CLASS_LABELS, fontsize=9)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Producer's & User's Accuracy", fontsize=11)
ax.legend(handles=legend_handles, fontsize=7, loc="lower right")
ax.grid(True, axis="y", alpha=0.3)

fig.tight_layout()
out_path = Path(__file__).parent / "forest_metrics_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_path}")
