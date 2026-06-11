"""Logging configuration: bare messages on stdout + timestamped per-run files.

Project rule: use logging, never print. Handlers live on the ROOT logger on
purpose, so module-level logs from seabed_tiler (rotation, splits, ...) are
captured alongside seabed_unet's own — print can't be filtered or persisted,
and unconfigured module loggers silently drop their messages.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_STDOUT_FLAG = "_seabed_stdout_handler"


def setup_logging(level: int = logging.INFO) -> None:
    """Idempotent: attach one bare-format stdout handler to the root logger."""
    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(h, _STDOUT_FLAG, False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        setattr(handler, _STDOUT_FLAG, True)
        root.addHandler(handler)


def add_file_handler(path: Path) -> logging.Handler:
    """Mirror all log records to ``path`` (timestamped format). Caller removes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    logging.getLogger().addHandler(handler)
    return handler


def remove_handler(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()
