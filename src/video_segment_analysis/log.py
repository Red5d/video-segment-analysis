"""Logging setup: console (rich handler when available) + optional file."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False


def setup_logging(level: str = "INFO", file: str | Path | None = None) -> None:
    global _CONFIGURED
    root = logging.getLogger("vsa")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if _CONFIGURED:
        if file:
            _add_file_handler(root, Path(file))
        return

    try:
        from rich.logging import RichHandler
        handler: logging.Handler = RichHandler(
            show_path=False, log_time_format="[%H:%M:%S]")
        formatter = logging.Formatter("%(name)s: %(message)s")
        handler.setFormatter(formatter)
    except ImportError:  # pragma: no cover
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
    root.addHandler(handler)
    root.propagate = False

    if file:
        _add_file_handler(root, Path(file))
    _CONFIGURED = True


def _add_file_handler(root: logging.Logger, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
    root.addHandler(fh)
