"""seabed_unet — lightweight U-Net training on the seabed_tiler tile dataset.

Consumes the rotation-aware tile runs under outputs/<polygon>/<run_tag>_rot
(base train/val/test tiles) and _rotaug (train-only augmentation passes), per
the binding contract in docs/DATA_AUGMENTATION.md.
"""

__version__ = "0.1.0"
