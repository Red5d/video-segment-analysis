"""Fake ASR client for tests and ``--dry-run`` plumbing (DESIGN §8.2).

Returns canned ``Utterance`` lists (constructor, JSON fixture, or a
deterministic default) time-scaled to the duration of the submitted audio.
"""

from __future__ import annotations

import contextlib
import json
import logging
import subprocess
import wave
from dataclasses import replace
from pathlib import Path

from .base import AsrClient, Utterance, Word

logger = logging.getLogger("vsa.asr.fake")

_DEFAULT_SENTENCES = [
    "This is the first sentence of the canned transcript.",
    "The second sentence continues the fake talk.",
    "And this final sentence closes out the segment.",
]


def probe_duration(path: Path) -> float:
    """Best-effort audio duration in seconds (ffmpegio → ffprobe → wave)."""
    try:  # lazy: ffmpegio may not exist yet / may shell out
        from ..ffmpegio import probe
        info = probe(path)
        dur = info.get("duration") if isinstance(info, dict) else getattr(info, "duration", None)
        if dur:
            return float(dur)
    except Exception:  # noqa: BLE001 — best-effort chain
        pass
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", str(path)],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout.strip()
        return float(out)
    except Exception:  # noqa: BLE001
        pass
    try:
        with contextlib.closing(wave.open(str(path))) as w:
            rate = w.getframerate()
            if rate:
                return w.getnframes() / rate
    except Exception:  # noqa: BLE001
        pass
    logger.warning("could not probe duration of %s; assuming 30 s", path)
    return 30.0


def _spread(texts: list[str], duration: float) -> list[Utterance]:
    """Deterministic default: sentences spread evenly over ``duration``,
    with proportional word timestamps."""
    n = len(texts)
    out: list[Utterance] = []
    for i, text in enumerate(texts):
        start = duration * i / n
        end = duration * (i + 1) / n
        tokens = text.split()
        chars = sum(len(t) for t in tokens) or 1
        words, cursor = [], start
        span = max(end - start, 0.0)
        for tok in tokens:
            w = span * len(tok) / chars
            words.append(Word(tok, cursor, cursor + w))
            cursor += w
        out.append(Utterance(start, end, text, words))
    return out


def _load_fixture(path: Path) -> list[Utterance]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for item in data:
        words = [Word(**w) for w in item.get("words") or []] or None
        out.append(Utterance(start=float(item["start"]), end=float(item["end"]),
                             text=str(item["text"]), words=words))
    return out


def _scale(u: Utterance, k: float) -> Utterance:
    words = ([Word(w.text, w.start * k, w.end * k) for w in u.words]
             if u.words is not None else None)
    return replace(u, start=u.start * k, end=u.end * k, words=words)


class FakeAsrClient(AsrClient):
    """Canned transcriptions scaled to each submitted audio's duration."""

    def __init__(self, canned: list[Utterance] | None = None,
                 fixture_path: Path | None = None) -> None:
        self.canned = list(canned) if canned is not None else None
        if fixture_path is not None:
            self.canned = _load_fixture(Path(fixture_path))

    def transcribe(self, audio_path: Path, language: str | None = None) -> list[Utterance]:
        duration = probe_duration(Path(audio_path))
        if self.canned is None:
            return _spread(_DEFAULT_SENTENCES, duration)
        span = max((u.end for u in self.canned), default=0.0)
        k = duration / span if span > 0 else 1.0
        return [_scale(u, k) for u in self.canned]
