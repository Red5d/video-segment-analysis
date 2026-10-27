"""Silence → speech detection with ffmpeg's ``silencedetect`` filter (DESIGN §6).

:func:`parse_silencedetect` is pure (unit-tested against captured stderr
fixtures); :func:`detect_silences` runs the filter;
:func:`silences_to_speech` keeps the interval inversion/merge logic here so
the pipeline orchestration stays thin.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

from . import ffmpegio, timeline
from .config import SegmentationConfig, VadConfig
from .errors import SourceError, MediaError
from .timeline import Interval

logger = logging.getLogger("vsa.vad")

_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?\d+(?:\.\d+)?)")
_SILENCE_END_RE = re.compile(
    r"silence_end:\s*(-?\d+(?:\.\d+)?)"
    r"(?:\s*\|\s*silence_duration:\s*(-?\d+(?:\.\d+)?))?")


def parse_silencedetect(stderr: str) -> list[Interval]:
    """Parse ``silence_start`` / ``silence_end`` lines from silencedetect
    output (DESIGN §6.2).

    Tolerates interleaved noise lines (progress, other filters). A trailing
    open-ended ``silence_start`` (silence runs to EOF) is emitted with
    ``end = math.inf`` — the caller closes it at the media duration. A
    ``silence_start`` arriving while one is still pending (stream
    discontinuity) closes the pending one at the new start.
    """
    out: list[Interval] = []
    pending: float | None = None
    for line in stderr.splitlines():  # splitlines() absorbs CRLF
        start_m = _SILENCE_START_RE.search(line)
        if start_m:
            start = float(start_m.group(1))
            if pending is not None:
                out.append((pending, start))
            pending = start
            continue
        end_m = _SILENCE_END_RE.search(line)
        if end_m:
            end = float(end_m.group(1))
            if pending is None:
                continue  # unmatched end (noise) — ignore
            out.append((pending, end))
            pending = None
    if pending is not None:
        out.append((pending, math.inf))  # caller clamps to media duration
    return timeline.normalize(out)


def detect_silences(wav_path: Path, cfg: VadConfig, total_duration: float) -> list[Interval]:
    """Run ``silencedetect`` on the 16 kHz mono WAV and return silence
    intervals (normalized, trailing silence closed at ``total_duration``).

    Raises :class:`SourceError` with tuning guidance when *zero* silences
    are found — almost always a mis-tuned ``noise_db`` /
    ``min_silence_duration`` (DESIGN §6.2).
    """
    cmd = [ffmpegio.FFMPEG, "-hide_banner", "-nostats", "-i", str(wav_path),
           "-af", f"silencedetect=noise={cfg.noise_db}dB:d={cfg.min_silence_duration:g}",
           "-f", "null", "-"]
    logger.info("silence-detect: %s (noise=%ddB, d=%gs, duration=%.1fs)",
                wav_path.name, cfg.noise_db, cfg.min_silence_duration, total_duration)
    try:
        stderr = ffmpegio.run(cmd, merge_stderr=True)
    except MediaError as exc:
        raise SourceError(f"silencedetect failed on {wav_path}: {exc}") from exc

    parsed = parse_silencedetect(stderr)
    # Drop filter artifacts shorter than the requested minimum (§6.2),
    # close open-ended trailing silence at the file duration.
    silences = [(s, min(e, total_duration)) for s, e in parsed if e - s >= cfg.min_silence_duration]
    silences = timeline.normalize(silences, total_duration=total_duration)
    if not silences:
        raise SourceError(
            f"silencedetect found zero silences in {wav_path.name} "
            f"(noise_db={cfg.noise_db}, min_silence_duration={cfg.min_silence_duration}). "
            "The audio is likely louder than the silence threshold: raise "
            "`vad.noise_db` toward 0 (e.g. -30 for rooms with applause/floor "
            "noise) and/or increase `vad.min_silence_duration`. "
            "Verify the file actually contains audio (ffprobe the WAV)."
        )
    logger.info("silence-detect: %d silences (%.1fs total)",
                len(silences), timeline.total(silences))
    return silences


def silences_to_speech(silences: list[Interval], total_duration: float,
                       cfg_sil: VadConfig, cfg_seg: SegmentationConfig) -> list[Interval]:
    """Invert silence intervals into speech intervals and merge runs
    separated by less than ``segmentation.merge_speech_gap`` (§6.3).

    ``cfg_sil`` is kept in the signature for engine parity (reserved for a
    future ``silero`` engine that needs its own thresholds); the ffmpeg
    inversion path does not use it.
    """
    speech = timeline.invert(silences, total_duration)
    merged = timeline.merge(speech, cfg_seg.merge_speech_gap)
    logger.info("speech intervals: %d (merged with gap<%gs), %.1fs of speech",
                len(merged), cfg_seg.merge_speech_gap, timeline.total(merged))
    return merged
