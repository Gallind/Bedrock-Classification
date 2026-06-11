"""
Render .xyz seabed survey files (X, Y, Z) to georeferenced images using
numpy + matplotlib.  Works for any regular-grid XYZ dataset in this project.
"""

import logging
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

logger = logging.getLogger("render_python")

DB = "DataBase"
DATASETS = [
    {
        "xyz":    f"{DB}/bathymetry Grid 0.5m.xyz",
        "output": "bathymetry_python.png",
        "title":  "Bathymetry – 0.5 m grid (UTM 36N)",
        "cmap":   "Blues_r",
        "label":  "Depth (m)",
    },
    {
        "xyz":    f"{DB}/Slope  Grid 0.5m.xyz",
        "output": "slope_python.png",
        "title":  "Slope – 0.5 m grid (UTM 36N)",
        "cmap":   "YlOrRd",
        "label":  "Slope (°)",
    },
]


def xyz_to_image(xyz_path, output_path, title, cmap, label, pixel_size=0.5):
    logger.info(f"\n[+] Reading {os.path.basename(xyz_path)} …")
    df = pd.read_csv(xyz_path, header=None, names=["x", "y", "z"], dtype=np.float64)
    logger.info(f"    {len(df):,} points  |  Z range: {df.z.min():.3f} → {df.z.max():.3f}")

    # Snap coordinates to a uniform grid origin
    x_min = df.x.min()
    y_max = df.y.max()   # top-left origin (row 0 = northernmost)

    cols = np.round((df.x.values - x_min) / pixel_size).astype(np.int32)
    rows = np.round((y_max - df.y.values) / pixel_size).astype(np.int32)

    n_rows = rows.max() + 1
    n_cols = cols.max() + 1
    logger.info(f"    Grid: {n_rows} rows × {n_cols} cols")

    Z = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    Z[rows, cols] = df.z.values

    # Clip colour range to 2nd–98th percentile to avoid outlier distortion
    vmin, vmax = np.nanpercentile(Z, [2, 98])

    fig, ax = plt.subplots(figsize=(12, 10), dpi=150)

    x_max = df.x.max()
    y_min = df.y.min()
    img = ax.imshow(
        Z,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        aspect="equal",
        origin="upper",
        extent=[x_min, x_max, y_min, y_max],
        interpolation="nearest",
    )

    cbar = fig.colorbar(img, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, fontsize=11)

    ax.set_title(title, fontsize=13, pad=10)
    ax.set_xlabel("Easting (m)", fontsize=10)
    ax.set_ylabel("Northing (m)", fontsize=10)
    ax.ticklabel_format(style="plain", axis="both")
    ax.tick_params(labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"    Saved → {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
    for ds in DATASETS:
        xyz_to_image(
            ds["xyz"],
            ds["output"],
            ds["title"],
            ds["cmap"],
            ds["label"],
        )
    logger.info("\nDone.")
