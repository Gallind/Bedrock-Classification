"""Read .xyz point grids and snap them onto the master raster grid.

The snapping logic mirrors scripts/render_python.py (the project's existing, validated
XYZ renderer): map each point to the nearest cell of a regular grid anchored at the
master grid's top-left origin. Cells with no point become ``nodata`` — this is how the
sparse (~50 % coverage) bathymetry/slope surveys are represented honestly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_xyz_df(path: str | Path) -> pd.DataFrame:
    """Load a comma-separated X,Y,Z file (no header) into a DataFrame."""
    return pd.read_csv(path, header=None, names=["x", "y", "z"], dtype=np.float64)


def grid_xyz(
    df: pd.DataFrame,
    x_min: float,
    y_max: float,
    n_rows: int,
    n_cols: int,
    res: float,
    nodata: float,
) -> np.ndarray:
    """Place XYZ points onto an ``(n_rows, n_cols)`` grid anchored at (x_min, y_max).

    Points falling outside the grid are dropped; empty cells are filled with ``nodata``.
    """
    cols = np.round((df.x.values - x_min) / res).astype(np.int64)
    rows = np.round((y_max - df.y.values) / res).astype(np.int64)

    arr = np.full((n_rows, n_cols), nodata, dtype=np.float32)
    inside = (cols >= 0) & (cols < n_cols) & (rows >= 0) & (rows < n_rows)
    arr[rows[inside], cols[inside]] = df.z.values[inside].astype(np.float32)
    return arr
