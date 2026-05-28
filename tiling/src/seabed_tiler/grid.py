"""Build the overlapping tile grid in world (UTM meter) coordinates.

The grid is defined once here and every layer is cut against the same windows, which
is what keeps features and labels co-registered regardless of their source resolution.
Pure functions only — no I/O — so this is cheap to unit-test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TileWindow:
    """A single tile's position (grid index) and bounds (UTM meters, top-left origin)."""

    row: int
    col: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def build_windows(
    extent: tuple[float, float, float, float],
    tile_size_m: float,
    stride_m: float,
    keep_partial_edge: bool = False,
    eps: float = 1e-6,
) -> list[TileWindow]:
    """Tile ``extent`` (xmin, ymin, xmax, ymax) into overlapping squares.

    Rows increase southward (row 0 at the top, ymax), columns increase eastward, matching
    raster row/col convention. With ``stride_m == tile_size_m / 2`` neighbouring tiles
    share exactly 50 % of their area.

    ``keep_partial_edge`` keeps trailing tiles that extend past the extent; otherwise only
    fully-contained tiles are emitted.
    """
    if tile_size_m <= 0 or stride_m <= 0:
        raise ValueError("tile_size_m and stride_m must be positive")

    xmin, ymin, xmax, ymax = extent
    windows: list[TileWindow] = []

    row = 0
    while True:
        ytop = ymax - row * stride_m
        ybot = ytop - tile_size_m
        if keep_partial_edge:
            if ytop <= ymin + eps:
                break
        elif ybot < ymin - eps:
            break

        col = 0
        while True:
            x0 = xmin + col * stride_m
            x1 = x0 + tile_size_m
            if keep_partial_edge:
                if x0 >= xmax - eps:
                    break
            elif x1 > xmax + eps:
                break

            windows.append(TileWindow(row, col, x0, ybot, x1, ytop))
            col += 1
        row += 1

    return windows
