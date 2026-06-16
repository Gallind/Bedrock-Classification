"""seabed_forest — per-pixel tree baseline (RF + HistGradientBoosting) on the
seabed_tiler tile dataset. Reuses seabed_unet's torch-free data/normalize/metrics/
splits/crossval/config; never modifies that package. See
docs/superpowers/specs/2026-06-15-perpixel-tree-baseline-design.md.
"""

__version__ = "0.1.0"
