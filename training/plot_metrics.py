"""Plot train loss, val macro-Dice, and test macro-Dice for unet_2band and unet_3band.

One subplot per experiment: train loss on left y-axis, val+test dice on right y-axis.
Test dice shown as a horizontal dashed line so all three series share the same x-axis.
"""

import json
import csv
from pathlib import Path
import matplotlib.pyplot as plt

RUNS_DIR = Path(__file__).parent / "runs"

EXPERIMENTS = {
    "unet_2band": {"label": "2-band (Bath+BS)", "loss_color": "#4C9BE8", "dice_color": "#1A5F9C"},
    "unet_3band": {"label": "3-band (Bath+BS+Slope)", "loss_color": "#E8834C", "dice_color": "#A84E14"},
}


def load_history(run_dir: Path) -> dict[str, list]:
    rows = {"epoch": [], "train_loss": [], "val_macro_dice": []}
    with open(run_dir / "history.csv") as f:
        for row in csv.DictReader(f):
            rows["epoch"].append(int(row["epoch"]))
            rows["train_loss"].append(float(row["train_loss"]))
            rows["val_macro_dice"].append(float(row["val_macro_dice"]))
    return rows


def load_test_dice(run_dir: Path) -> float:
    with open(run_dir / "eval_test" / "metrics.json") as f:
        return json.load(f)["macro_dice"]


fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("U-Net: Train Loss + Val/Test Macro-Dice", fontsize=14, fontweight="bold", y=1.01)

for ax, (exp_name, meta) in zip(axes, EXPERIMENTS.items()):
    h = load_history(RUNS_DIR / exp_name)
    test_dice = load_test_dice(RUNS_DIR / exp_name)
    epochs = h["epoch"]

    # ── left y-axis: train loss ──────────────────────────────────────────────
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss", color=meta["loss_color"])
    l_loss, = ax.plot(epochs, h["train_loss"], color=meta["loss_color"],
                      linewidth=1.8, label="Train Loss")
    ax.tick_params(axis="y", labelcolor=meta["loss_color"])
    ax.set_xlim(left=1)

    # ── right y-axis: val dice + test dice ───────────────────────────────────
    ax2 = ax.twinx()
    ax2.set_ylabel("Macro Dice", color=meta["dice_color"])
    l_val, = ax2.plot(epochs, h["val_macro_dice"], color=meta["dice_color"],
                      linewidth=1.8, alpha=0.85, label="Val Macro-Dice")
    l_test = ax2.axhline(test_dice, color=meta["dice_color"], linewidth=2,
                         linestyle="--", label=f"Test Macro-Dice ({test_dice:.4f})")
    ax2.tick_params(axis="y", labelcolor=meta["dice_color"])
    ax2.set_ylim(0, 1)

    # star on best val dice
    best_idx = max(range(len(h["val_macro_dice"])), key=lambda i: h["val_macro_dice"][i])
    ax2.scatter(epochs[best_idx], h["val_macro_dice"][best_idx],
                color=meta["dice_color"], s=90, zorder=5, marker="*")
    ax2.annotate(
        f"best val {h['val_macro_dice'][best_idx]:.3f}",
        (epochs[best_idx], h["val_macro_dice"][best_idx]),
        textcoords="offset points", xytext=(6, 4), fontsize=7.5,
        color=meta["dice_color"],
    )

    ax.set_title(meta["label"], fontsize=11)
    ax.grid(True, alpha=0.25)

    lines = [l_loss, l_val, l_test]
    ax.legend(lines, [l.get_label() for l in lines], loc="upper right", fontsize=8, framealpha=0.9)

fig.tight_layout()
out_path = Path(__file__).parent / "metrics_overview.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_path}")
