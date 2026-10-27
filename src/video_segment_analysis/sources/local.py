"""Local media file source (DESIGN §7.2/§7.3)."""

from __future__ import annotations

import logging
from pathlib import Path

from .. import ffmpegio
from ..config import Config
from ..errors import MediaError, SourceError
from ..ffmpegio import MediaInfo
from ..manifest import Segment
from .base import CutResult, SourceAdapter

logger = logging.getLogger("vsa.sources.local")

# Stream-copy cuts whose duration deviates from the request by more than
# this many seconds indicate sparse keyframes → recommend re-encoding (§7.3).
COPY_DURATION_TOLERANCE = 3.0


class LocalFileSource(SourceAdapter):
    """Cuts a file on disk with ffmpeg: ``local-copy`` (default) or
    ``local-encode``."""

    def __init__(self, path: Path, cfg: Config, workdir: Path) -> None:
        super().__init__(cfg, workdir)
        self.path = Path(path)

    def probe(self) -> MediaInfo:
        if not self.path.is_file():
            raise SourceError(f"local source not found: {self.path}")
        try:
            return ffmpegio.probe(self.path)
        except MediaError as exc:
            raise SourceError(f"cannot probe local source {self.path}: {exc}") from exc

    def extract_full_audio(self, dst_wav: Path) -> None:
        ffmpegio.extract_audio(self.path, dst_wav, self.cfg.audio)

    def _audio_source(self) -> Path:
        """Prefer the shared full WAV (already decoded); fall back to
        decoding the segment audio straight from the source file."""
        full_wav = self.workdir / "audio.wav"
        return full_wav if full_wav.is_file() else self.path

    def cut_segment(self, seg: Segment, dst_video: Path, dst_audio_wav: Path,
                    strategy: str) -> CutResult:
        strategy = "local-copy" if strategy in ("auto", "local-copy") else strategy
        if strategy == "yt-sections":
            raise SourceError(
                f"strategy yt-sections requires a URL source, not the local file {self.path}")
        if strategy not in ("local-copy", "local-encode"):
            raise SourceError(f"unknown cut strategy for a local source: {strategy}")

        start, end = seg.start, seg.end
        requested = end - start
        info = self.probe()

        # --- video ---------------------------------------------------------
        video: Path | None = None
        actual_start = start
        actual_end = end
        reencode_recommended = False
        if info.has_video:
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            logger.info("[cut:%s] %s %.3f-%.3f", seg.id, strategy, start, end)
            if strategy == "local-copy":
                ffmpegio.cut_video_copy(self.path, dst_video, start, end)
            else:
                ffmpegio.cut_video_encode(self.path, dst_video, start, end)
            video = dst_video
            # With `-ss` before `-i` + stream copy, the file starts at the
            # keyframe at/just before ``start``; ffprobe of the output
            # starts at 0 (make_zero), so the only observable drift is the
            # missing head:  actual_start = start - (requested - out_dur).
            # For re-encoded cuts the drift is ~0 (keyframe forced at cut).
            out_dur = ffmpegio.probe(dst_video).duration
            actual_start = start - max(0.0, requested - out_dur)
            actual_end = actual_start + out_dur
            if strategy == "local-copy" and abs(out_dur - requested) > COPY_DURATION_TOLERANCE:
                reencode_recommended = True
                logger.warning(
                    "[cut:%s] stream-copy cut is %.2fs vs requested %.2fs "
                    "(sparse keyframes) — re-run with strategy local-encode "
                    "and --force for this segment", seg.id, out_dur, requested)
        else:
            logger.info("[cut:%s] source has no video stream; skipping video cut", seg.id)

        # --- audio (always sample-accurate from the full 16k WAV) ----------
        dst_audio_wav.parent.mkdir(parents=True, exist_ok=True)
        ffmpegio.cut_audio_wav(self._audio_source(), dst_audio_wav, start, end,
                               self.cfg.audio)
        audio_dur = ffmpegio.probe(dst_audio_wav).duration
        if not info.has_video:  # audio-only source: trust the WAV measurement
            actual_start, actual_end = start, start + audio_dur

        return CutResult(actual_start=actual_start, actual_end=actual_end,
                         video=video, audio=dst_audio_wav, strategy=strategy,
                         reencode_recommended=reencode_recommended)

    def describe_source(self) -> dict:
        info = self.probe()
        return {
            "type": "local",
            "url": None,
            "path": str(self.path),
            "title": self.path.stem,
            "duration": info.duration,
        }
