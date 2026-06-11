"""Project-wide logging configuration (canonical home; stdlib-only).

Project rule: use logging, never print. Every module gets its own named logger
(`logger = logging.getLogger(__name__)`), so each record carries the module
name on both sinks. Handlers live on the ROOT logger on purpose, so logs from
seabed_tiler and seabed_unet are captured together — print can't be filtered
or persisted, and unconfigured module loggers silently drop their messages.

Per-command log files (attached by each CLI's main):
  seabed_tiler           -> <out_dir>/tiling.log
  seabed_tiler.to_jpg    -> <run_dir>/to_jpg.log
  seabed_tiler.stitch    -> <run_dir>/stitch.log
  seabed_unet.train      -> <run_dir>/train.log
  seabed_unet.evaluate   -> <run_dir>/eval_<split>.log
  seabed_unet.predict    -> <run_dir>/predict.log
  seabed_unet.crossval   -> <lopo_dir>/crossval.log (whole sweep, incl. folds)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_STDOUT_FLAG = "_seabed_stdout_handler"

STDOUT_FORMAT = "%(name)s: %(message)s"
FILE_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Idempotent: attach one compact stdout handler to the root logger."""
    root = logging.getLogger()
    root.setLevel(level)
    if not any(getattr(h, _STDOUT_FLAG, False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(STDOUT_FORMAT))
        setattr(handler, _STDOUT_FLAG, True)
        root.addHandler(handler)


def add_file_handler(path: Path) -> logging.Handler:
    """Mirror all log records to ``path`` (timestamped format). Caller removes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path)
    handler.setFormatter(logging.Formatter(FILE_FORMAT))
    logging.getLogger().addHandler(handler)
    return handler


def remove_handler(handler: logging.Handler) -> None:
    logging.getLogger().removeHandler(handler)
    handler.close()
