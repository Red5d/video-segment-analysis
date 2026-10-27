"""Source adapters: local files and yt-dlp sites (DESIGN §7.2/§7.3)."""

from .base import CutResult, SourceAdapter, resolve_source
from .local import LocalFileSource

__all__ = ["CutResult", "SourceAdapter", "LocalFileSource", "resolve_source"]
