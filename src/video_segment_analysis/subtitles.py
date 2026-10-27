"""SRT/VTT subtitle writing and reading (DESIGN §8.5).

All cue times are **seconds**; writers format them as
``HH:MM:SS,mmm`` (SRT, CRLF, UTF-8 no BOM) or ``HH:MM:SS.mmm``
(VTT, ``WEBVTT`` header, LF). Readers tolerate both separators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import logging

logger = logging.getLogger("vsa.subtitles")


@dataclass
class Cue:
    start: float
    end: float
    text: str
    index: int = 0


# ---------------------------------------------------------------------------
# Time formatting

def _split_ms(t: float) -> tuple[int, int, int, int]:
    """Seconds → (hours, minutes, seconds, milliseconds), clamped ≥ 0."""
    ms = max(0, int(round(float(t) * 1000)))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return h, m, s, ms


def format_srt_time(t: float) -> str:
    h, m, s, ms = _split_ms(t)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_time(t: float) -> str:
    h, m, s, ms = _split_ms(t)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


# ---------------------------------------------------------------------------
# Text helpers

def wrap_text(text: str, width: int) -> str:
    """Greedy word wrap; words longer than ``width`` stay intact on own line."""
    if width <= 0:
        return text
    lines: list[str] = []
    cur = ""
    for word in text.split():
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Normalization

def normalize_cues(
    cues: list[Cue],
    *,
    segment_duration: float | None = None,
    min_gap: float = 0.02,
    min_duration: float = 0.2,
    max_width: int | None = None,
) -> list[Cue]:
    """Sort, drop empty/degenerate cues, clamp, enforce non-overlap, renumber.

    Overlaps are resolved by shrinking the earlier cue's end so that
    ``prev.end <= next.start - min_gap``; a cue that would collapse
    (end < start + ``min_duration``) is dropped. When ``max_width`` is given
    the text is word-wrapped to that width.
    """
    items: list[Cue] = []
    for c in cues:
        text = " ".join((c.text or "").split())
        if not text:
            continue
        s, e = float(c.start), float(c.end)
        if segment_duration is not None:
            s = max(0.0, s)
            e = min(e, float(segment_duration))
        if e - s < min_duration:  # degenerate (also covers end <= start)
            continue
        items.append(replace(c, start=s, end=e, text=text))
    items.sort(key=lambda c: (c.start, c.end))

    out: list[Cue] = []
    for c in items:
        while out:
            prev = out[-1]
            allowed = c.start - min_gap
            if prev.end <= allowed:
                break
            if allowed - prev.start < min_duration:
                out.pop()  # shrinking would collapse prev → drop it
                continue
            prev.end = allowed
            break
        out.append(c)

    for i, c in enumerate(out, start=1):
        c.index = i
        if max_width is not None:
            c.text = wrap_text(c.text, max_width)
    if len(out) != len(cues):
        logger.debug("normalize_cues: %d -> %d cues", len(cues), len(out))
    return out


# ---------------------------------------------------------------------------
# Writers

def _render(cues: list[Cue], fmt: str) -> str:
    fmt_time = format_srt_time if fmt == "srt" else format_vtt_time
    blocks = [
        f"{c.index}\n{fmt_time(c.start)} --> {fmt_time(c.end)}\n{c.text}"
        for c in cues
    ]
    body = "\n\n".join(blocks)
    if fmt == "vtt":
        return "WEBVTT\n\n" + (body + "\n" if blocks else "")
    text = body + "\n" if blocks else ""
    return text.replace("\n", "\r\n")  # SRT: uniform CRLF, incl. text lines


def write_srt(path: str | Path, cues: list[Cue]) -> None:
    """UTF-8 (no BOM), CRLF, blank line between cues, 1-based indices."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_render(cues, "srt").encode("utf-8"))


def write_vtt(path: str | Path, cues: list[Cue]) -> None:
    """``WEBVTT`` header, dot millisecond separator, LF line endings."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(_render(cues, "vtt").encode("utf-8"))


# ---------------------------------------------------------------------------
# Readers

_TIME_RE = re.compile(
    r"(?:(\d{1,3}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})\s*-->\s*"
    r"(?:(\d{1,3}):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
)


def _secs(h: str | None, m: str, s: str, ms: str) -> float:
    return int(h or 0) * 3600 + int(m) * 60 + int(s) + int(ms.ljust(3, "0")) / 1000.0


def _parse_blocks(text: str) -> list[Cue]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    cues: list[Cue] = []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or block.startswith(("WEBVTT", "NOTE", "STYLE", "REGION")):
            continue
        lines = block.split("\n")
        ti = next((i for i, ln in enumerate(lines) if _TIME_RE.search(ln)), None)
        if ti is None:
            continue
        m = _TIME_RE.search(lines[ti])
        assert m is not None
        start = _secs(m.group(1), m.group(2), m.group(3), m.group(4))
        end = _secs(m.group(5), m.group(6), m.group(7), m.group(8))
        index = 0
        if ti == 1 and lines[0].strip().isdigit():
            index = int(lines[0].strip())
        body = "\n".join(ln.strip() for ln in lines[ti + 1:]).strip()
        if body:
            cues.append(Cue(start=start, end=end, text=body, index=index))
    for i, c in enumerate(cues, start=1):
        if not c.index:
            c.index = i
    return cues


def read_srt(path: str | Path) -> list[Cue]:
    return _parse_blocks(Path(path).read_text(encoding="utf-8-sig"))


def read_vtt(path: str | Path) -> list[Cue]:
    return _parse_blocks(Path(path).read_text(encoding="utf-8-sig"))
