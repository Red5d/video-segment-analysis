"""YouTube / yt-dlp source adapter (DESIGN §6.1, §7.3).

Audio for VAD/ASR always comes from ONE full-audio download (cheap,
robust). Video cutting defaults to *full download once* + local cuts;
``yt-sections`` is the opt-in range-download path.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import ffmpegio
from ..config import Config
from ..errors import SourceError
from ..ffmpegio import MediaInfo
from ..manifest import Segment
from ..retry import retry_with_backoff
from .base import CutResult, SourceAdapter

logger = logging.getLogger("vsa.sources.youtube")


def section_range(start: float, end: float) -> str:
    """yt-dlp ``download_ranges`` string: ``*H:MM:SS-H:MM:SS`` (leading ``*``
    marks an absolute timeline range; hours unpadded, mm/ss zero-padded)."""
    def hms(t: float) -> str:
        t = max(0, int(round(t)))
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"*{hms(start)}-{hms(end)}"


class YoutubeSource(SourceAdapter):
    """Any yt-dlp-supported URL. Uses the yt-dlp **Python API** only."""

    def __init__(self, url: str, cfg: Config, workdir: Path) -> None:
        super().__init__(cfg, workdir)
        self.url = url
        self.full_video: Path | None = None  # set by pipeline after full download
        self._info: dict | None = None

    # -- yt-dlp plumbing ----------------------------------------------------

    def _base_opts(self) -> dict:
        opts: dict = {
            "noplaylist": True,
            "quiet": True,
            "no_warnings": False,
            "retries": 3,
            "fragment_retries": 10,
            "concurrent_fragment_downloads": 4,
            "nocheckcertificate": False,
        }
        if self.cfg.source.cookiefile:
            opts["cookiefile"] = self.cfg.source.cookiefile
        return opts

    def _ydl(self, extra: dict | None = None):
        import yt_dlp  # lazy: keeps unit tests import-light

        opts = self._base_opts()
        opts.update(extra or {})
        return yt_dlp.YoutubeDL(opts)

    def _extract_info(self) -> dict:
        if self._info is None:
            try:
                with self._ydl() as ydl:
                    self._info = ydl.extract_info(self.url, download=False)
            except Exception as exc:  # yt_dlp.utils.DownloadError et al.
                raise SourceError(
                    f"yt-dlp could not read {self.url}: {exc}. If this is an "
                    "age/region-gated stream pass source.cookiefile.") from exc
            if self._info is None:
                raise SourceError(f"yt-dlp returned no metadata for {self.url}")
        return self._info

    # -- SourceAdapter -------------------------------------------------------

    def probe(self) -> MediaInfo:
        info = self._extract_info()
        duration = float(info.get("duration") or 0.0)
        if duration <= 0:
            raise SourceError(
                f"yt-dlp reported no duration for {self.url} — for live "
                "streams the broadcast must have ended before processing.")
        return MediaInfo(duration=duration, has_video=bool(info.get("formats")),
                         has_audio=True, width=info.get("width"),
                         height=info.get("height"), vcodec=None, acodec=None)

    def extract_full_audio(self, dst_wav: Path) -> None:
        """One full-audio download → 16 kHz WAV (DESIGN §6.1)."""
        dst_wav = Path(dst_wav)
        dst_wav.parent.mkdir(parents=True, exist_ok=True)
        outstem = dst_wav.with_suffix("")
        opts = {
            "format": "bestaudio/best",
            "postprocessors": [{"key": "FFmpegExtractAudio",
                                "preferredcodec": "wav",
                                "preferredquality": "0"}],
            "outtmpl": str(outstem) + ".%(ext)s",
        }
        logger.info("[yt] downloading full audio: %s", self.url)

        def _go() -> None:
            with self._ydl(opts) as ydl:
                ydl.download([self.url])

        retry_with_backoff(_go, retries=2, backoff=(5, 15, 60), label="yt-audio")
        produced = dst_wav if dst_wav.is_file() else _newest_with_stem(outstem)
        if produced is None:
            raise SourceError(f"yt-dlp produced no audio file for {self.url}")
        if produced != dst_wav:
            produced.replace(dst_wav)
        logger.info("[yt] full audio ready: %s", dst_wav)

    def download_full_video(self, dst_mp4: Path, video_format: str) -> Path:
        """Download the full media once (§7.3 auto behavior for YouTube)."""
        dst_mp4 = Path(dst_mp4)
        dst_mp4.parent.mkdir(parents=True, exist_ok=True)
        opts = {
            "format": video_format,
            "merge_output_format": "mp4",
            "outtmpl": str(dst_mp4.with_suffix("")) + ".%(ext)s",
        }
        logger.info("[yt] downloading full video (%s): %s", video_format, self.url)

        def _go() -> None:
            with self._ydl(opts) as ydl:
                ydl.download([self.url])

        retry_with_backoff(_go, retries=2, backoff=(10, 30, 120), label="yt-video")
        produced = dst_mp4 if dst_mp4.is_file() else _newest_with_stem(dst_mp4.with_suffix(""))
        if produced is None:
            raise SourceError(f"yt-dlp produced no video file for {self.url}")
        if produced != dst_mp4:
            produced.replace(dst_mp4)
        self.full_video = dst_mp4
        return dst_mp4

    def cut_segment(self, seg: Segment, dst_video: Path, dst_audio_wav: Path,
                    strategy: str) -> CutResult:
        start, end = seg.start, seg.end
        requested = end - start

        # --- video ----------------------------------------------------------
        if strategy == "local-encode" and self.full_video:
            # frame-accurate re-encode cut (repair path for copy-cut drift)
            ffmpegio.cut_video_encode(self.full_video, dst_video, start, end)
            strat_used = "local-encode"
            out_dur = ffmpegio.probe(dst_video).duration
            actual_start = start - max(0.0, requested - out_dur)
            reencode = False  # already an encode; no further recommendation
        elif strategy in ("auto", "local-copy", "yt-sections") and self.full_video:
            # full video already downloaded → local cut (yt-sections falls
            # back here per segment: cheaper than a section refetch)
            ffmpegio.cut_video_copy(self.full_video, dst_video, start, end)
            strat_used = "local-copy"
            out_dur = ffmpegio.probe(dst_video).duration
            actual_start = start - max(0.0, requested - out_dur)
            reencode = abs(out_dur - requested) > 3.0
        elif strategy == "yt-sections":
            strat_used = "yt-sections"
            dst_video.parent.mkdir(parents=True, exist_ok=True)
            import yt_dlp
            opts = {
                "format": self.cfg.source.video_format,
                "merge_output_format": "mp4",
                "force_keyframes_at_cuts": True,
                "download_ranges": yt_dlp.utils.download_range_list(
                    None, [section_range(start, end)]),
                "outtmpl": str(dst_video.with_suffix("")) + ".%(ext)s",
            }

            def _go() -> None:
                with self._ydl(opts) as ydl:
                    ydl.download([self.url])

            retry_with_backoff(_go, retries=2, backoff=(10, 30, 120),
                               label=f"yt-section {seg.id}")
            produced = _newest_with_stem(dst_video.with_suffix(""))
            if produced is None:
                raise SourceError(f"yt-sections produced no file for {seg.id}")
            if produced != dst_video:
                produced.replace(dst_video)
            out_dur = ffmpegio.probe(dst_video).duration
            actual_start, reencode = start, False
        else:
            raise SourceError(
                "no full video downloaded — run `vsa segment` on this source "
                "first (YouTube auto downloads the full video once), or use "
                "--strategy yt-sections")
        actual_end = actual_start + out_dur

        # --- audio: always from the full WAV (§7.3) --------------------------
        full_wav = self.workdir / "audio.wav"
        if not full_wav.is_file():
            raise SourceError(f"missing {full_wav} — run extract_audio first")
        ffmpegio.cut_audio_wav(full_wav, dst_audio_wav, start, end, self.cfg.audio)

        return CutResult(actual_start=actual_start, actual_end=actual_end,
                         video=dst_video, audio=dst_audio_wav,
                         strategy=strat_used, reencode_recommended=reencode)

    def describe_source(self) -> dict:
        info = self._extract_info()
        return {
            "type": "youtube",
            "url": self.url,
            "path": str(self.full_video) if self.full_video else None,
            "title": info.get("title"),
            "duration": float(info.get("duration") or 0.0),
        }


def _newest_with_stem(stem: Path) -> Path | None:
    """``stem.<ext>`` candidates in the stem's directory, newest first."""
    cands = sorted(stem.parent.glob(stem.name + ".*"),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None
