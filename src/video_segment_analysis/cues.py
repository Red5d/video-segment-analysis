"""Cue builder: utterances → subtitle cues (DESIGN §8.4/§8.5).

Two timing paths, both on **segment-relative** times:
  * words present (aligner): greedy word→cue grouping — break at
    sentence-final punctuation, ``max_cue_duration``, ``max_cue_gap`` or
    ``max_cue_chars``; cue start/end are the first/last word times.
  * no words: one cue per utterance; oversized utterances split at
    sentence boundaries with time allocated proportionally by character
    count inside the utterance window (monotonic, never overlapping).
"""

from __future__ import annotations

import logging
import re

from .asr.base import Utterance, Word
from .config import SubtitlesConfig
from .subtitles import Cue, normalize_cues, wrap_text

logger = logging.getLogger("vsa.cues")

_SENTENCE_END = ".!?…"
_CLOSING = "”\"')»]}’"
_SENT_RE = re.compile(r".*?(?:[.!?…]+[" + re.escape(_CLOSING) + r"]*\s+|$)", flags=re.S)


def _ends_sentence(text: str) -> bool:
    t = text.strip().rstrip(_CLOSING)
    return bool(t) and t[-1] in _SENTENCE_END


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (0x3000 <= o <= 0x9FFF or 0xAC00 <= o <= 0xD7AF
            or 0xF900 <= o <= 0xFAFF or 0xFF00 <= o <= 0xFFEF)


def _join(tokens: list[str]) -> str:
    """Join aligner tokens: spaces between words, none around CJK chars."""
    out = ""
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if not out:
            out = tok
        elif _is_cjk(tok[0]) or _is_cjk(out[-1]):
            out += tok
        else:
            out += " " + tok
    return out


def _cues_from_words(words: list[Word], cfg: SubtitlesConfig) -> list[Cue]:
    cues: list[Cue] = []
    cur: list[Word] = []

    def flush() -> None:
        if cur:
            text = _join([w.text for w in cur])
            if text:
                cues.append(Cue(cur[0].start, cur[-1].end, text))
            cur.clear()

    for w in words:
        if not w.text.strip():
            continue
        if cur:
            gap = w.start - cur[-1].end
            duration = w.end - cur[0].start
            prospective = len(_join([x.text for x in cur] + [w.text]))
            if (gap > cfg.max_cue_gap or duration > cfg.max_cue_duration
                    or prospective > cfg.max_cue_chars):
                flush()
        cur.append(w)
        if _ends_sentence(w.text):
            flush()
    flush()
    return cues


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_RE.findall(text) if s.strip()]


def _split_text(text: str, max_chars: int) -> list[str]:
    """Sentences first, then long sentences wrapped at ``max_chars``."""
    parts: list[str] = []
    for sent in _split_sentences(text):
        parts.extend(wrap_text(sent, max_chars).split("\n"))
    return [p for p in (p.strip() for p in parts) if p]


def _cues_from_text(u: Utterance, cfg: SubtitlesConfig) -> list[Cue]:
    text = " ".join(u.text.split())
    span = u.end - u.start
    parts = _split_text(text, cfg.max_cue_chars)
    if len(parts) <= 1 or span <= 0:
        return [Cue(u.start, u.end, text)]
    chars = sum(len(p) for p in parts)
    cues: list[Cue] = []
    cursor = u.start
    for i, part in enumerate(parts):
        if i == len(parts) - 1:
            end = u.end  # exact window coverage
        else:
            end = min(cursor + span * len(part) / chars, u.end)
            end = max(end, cursor + span * 1e-3)  # keep non-degenerate
        cues.append(Cue(cursor, end, part))
        cursor = end
    return cues


def utterances_to_cues(utterances: list[Utterance], *, cfg: SubtitlesConfig,
                       segment_duration: float | None = None) -> list[Cue]:
    """Build normalized cues from segment-relative utterances."""
    cues: list[Cue] = []
    for u in utterances:
        if u.words:
            # text is derived from the words; an empty .text is fine
            cues.extend(_cues_from_words(u.words, cfg))
        elif u.text.strip():
            cues.extend(_cues_from_text(u, cfg))
    return normalize_cues(cues, segment_duration=segment_duration,
                          max_width=cfg.max_cue_chars)
