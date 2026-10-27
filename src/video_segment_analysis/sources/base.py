"""Source adapter ABC + resolution (DESIGN §7.2).

A source adapter knows how to probe its media, produce the single full
16 kHz WAV, and cut per-segment media. The pipeline talks only to this
interface; :func:`resolve_source` picks the implementation from the
``--source`` string and ``source.type`` config.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from ..config import Config
from ..errors import SourceError
from ..ffmpegio import MediaInfo
from ..manifest import Segment

logger = logging.getLogger("vsa.sources")


@dataclass
class CutResult:
    """Measured result of one segment cut.

    ``actual_start``/``actual_end`` are absolute source-timeline seconds
    measured from the produced file (stream-copy cuts start on an earlier
    keyframe — see ``LocalFileSource.cut_segment`` for the derivation).
    """

    actual_start: float
    actual_end: float
    video: Path | None
    audio: Path | None
    strategy: str
    reencode_recommended: bool = False


class SourceAdapter(ABC):
    """Interface every media source implements."""

    def __init__(self, cfg: Config, workdir: Path) -> None:
        self.cfg = cfg
        self.workdir = Path(workdir)

    @abstractmethod
    def probe(self) -> MediaInfo:
        """Media facts of the full source (duration, streams)."""

    @abstractmethod
    def extract_full_audio(self, dst_wav: Path) -> None:
        """Produce the single full 16 kHz mono WAV used by VAD/ASR (§6.1)."""

    @abstractmethod
    def cut_segment(self, seg: Segment, dst_video: Path, dst_audio_wav: Path,
                    strategy: str) -> CutResult:
        """Cut one segment's video + audio; report the measured timespan."""

    @abstractmethod
    def describe_source(self) -> dict:
        """Fields for ``manifest.SourceInfo`` (type/url/path/title/duration/...)."""


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://"))


def resolve_source(source: str, cfg: Config, workdir: Path) -> SourceAdapter:
    """Pick a :class:`SourceAdapter` for ``source`` (path or URL).

    ``auto``: an existing path on disk → local; an ``http(s)`` URL →
    youtube. ``source.type`` overrides must match the given string,
    otherwise :class:`SourceError` is raised.
    """
    from .local import LocalFileSource

    path = Path(source).expanduser()
    url = _is_url(source)
    exists = path.exists()
    stype = cfg.source.type

    if stype == "local":
        if url:
            raise SourceError(
                f"source.type=local but --source is a URL: {source!r} "
                "(use source.type=auto or youtube)")
        if not exists:
            raise SourceError(f"source.type=local but file not found: {source}")
        return LocalFileSource(path, cfg, workdir)

    if stype == "youtube":
        if not url:
            raise SourceError(
                f"source.type=youtube but --source is not an http(s) URL: {source!r}")
        from .youtube import YoutubeSource

        return YoutubeSource(source, cfg, workdir)

    # auto
    if exists:
        return LocalFileSource(path, cfg, workdir)
    if url:
        from .youtube import YoutubeSource

        return YoutubeSource(source, cfg, workdir)
    raise SourceError(
        f"cannot resolve source {source!r}: not an existing file and not an "
        "http(s):// URL (set source.type explicitly to force a backend)")
