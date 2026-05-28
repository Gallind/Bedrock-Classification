import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from seabed_tiler.grid import build_windows


def test_stride_is_half_tile_gives_50pct_overlap():
    # 100x100 m extent, 10 m tiles, 5 m stride.
    windows = build_windows((0, 0, 100, 100), tile_size_m=10, stride_m=5)
    by_index = {(w.row, w.col): w for w in windows}

    a = by_index[(0, 0)]
    right = by_index[(0, 1)]
    below = by_index[(1, 0)]

    # Neighbour origins differ by exactly the stride (5 m).
    assert right.xmin - a.xmin == 5
    assert a.ymax - below.ymax == 5

    # Horizontal overlap is half a tile.
    overlap_w = a.xmax - right.xmin
    assert overlap_w == 5
    assert overlap_w / (a.xmax - a.xmin) == 0.5


def test_full_tile_count_drop_partial_edge():
    # 100 m / 5 m stride, tile 10 m: last full tile starts at 90 -> origins 0..90 = 19 cols.
    windows = build_windows((0, 0, 100, 100), tile_size_m=10, stride_m=5)
    cols = max(w.col for w in windows) + 1
    rows = max(w.row for w in windows) + 1
    assert cols == 19
    assert rows == 19
    assert len(windows) == 19 * 19


def test_tiles_stay_within_extent_when_dropping_partials():
    windows = build_windows((0, 0, 95, 95), tile_size_m=10, stride_m=5)
    for w in windows:
        assert w.xmax <= 95 + 1e-6
        assert w.ymin >= 0 - 1e-6


def test_keep_partial_edge_adds_trailing_tiles():
    dropped = build_windows((0, 0, 100, 100), tile_size_m=10, stride_m=5)
    kept = build_windows((0, 0, 100, 100), tile_size_m=10, stride_m=5, keep_partial_edge=True)
    assert len(kept) > len(dropped)
