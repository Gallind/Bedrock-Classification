"""Spatially-blocked tile splits within polygons.

Tiles overlap 50%, so a random tile split puts near-duplicate pixels in both
train and test (forbidden by docs/DATA_AUGMENTATION.md). This module implements
the contract-compliant alternative: each polygon is cut into contiguous
train/val/test regions along its survey long axis, and every tile whose center
lies within ``buffer_m`` of a region boundary is dropped entirely.

Buffer guarantee: a 128 m tile extends at most half its diagonal (~90.5 m) from
its center in any direction (any rotation, including jittered augmentation
passes). With buffer_m = 128 dropped on BOTH sides of a boundary, centers of
kept tiles in different splits are >= 256 m apart along u > 181 m worst case
=> zero shared pixels across splits, guaranteed geometrically.

Augmented tiles inherit the region of their center and are kept only when that
region is train (contract: val/test regions are never augmented).
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .data import TileRecord

logger = logging.getLogger(__name__)

SPLIT_NAMES = ("train", "val", "test")


def assign_spatial_blocks(
    records: list[TileRecord],
    fractions: tuple[float, float, float],
    buffer_m: float,
) -> dict[str, list[TileRecord]]:
    """Assign records to train/val/test by position along each polygon's long axis.

    Per polygon: tile centers are projected onto the survey long axis
    (u = cx*cos(theta) + cy*sin(theta), theta = the polygon's BASE grid angle, so
    jittered augmentation tiles are measured in the same frame as base tiles).
    Region boundaries are u-quantiles of the BASE tiles at cumulative fractions,
    so the fractions refer to base-tile counts regardless of augmentation volume.
    """
    splits: dict[str, list[TileRecord]] = {name: [] for name in SPLIT_NAMES}
    by_polygon: dict[str, list[TileRecord]] = {}
    for r in records:
        by_polygon.setdefault(r.polygon, []).append(r)

    q1 = fractions[0]
    q2 = fractions[0] + fractions[1]

    for polygon, recs in by_polygon.items():
        base = [r for r in recs if not r.augmented]
        if not base:
            raise ValueError(f"{polygon}: no base (_rot) tiles to anchor the split")
        theta = math.radians(base[0].theta_deg)
        c, s = math.cos(theta), math.sin(theta)

        def u_of(r: TileRecord) -> float:
            return r.center_x * c + r.center_y * s

        base_u = np.array([u_of(r) for r in base])
        u_b1, u_b2 = np.quantile(base_u, [q1, q2])

        counts = {name: [0, 0] for name in SPLIT_NAMES}  # [kept, dropped]
        for r in recs:
            u = u_of(r)
            if abs(u - u_b1) < buffer_m or abs(u - u_b2) < buffer_m:
                continue  # buffer strip: dropped entirely, both sides
            region = "train" if u < u_b1 else ("val" if u < u_b2 else "test")
            if r.augmented and region != "train":
                counts[region][1] += 1
                continue  # augmented tiles never enter val/test
            splits[region].append(r)
            counts[region][0] += 1

        logger.info(
            "%s: theta=%.1f deg, boundaries u=(%.0f, %.0f), kept train=%d val=%d test=%d "
            "(aug dropped in val/test: %d)",
            polygon, base[0].theta_deg, u_b1, u_b2,
            counts["train"][0], counts["val"][0], counts["test"][0],
            counts["val"][1] + counts["test"][1],
        )

    for name in ("val", "test"):
        if not splits[name]:
            raise ValueError(
                f"spatial-block split produced an empty {name} set — "
                "reduce buffer_m or adjust fractions"
            )
    return splits
