"""Slide subtitle track builder (DESIGN §9.5).

Turns accepted slide-change times + VLM descriptions into the
``<seg>.slides.srt`` cue list:

- cue *i*: ``start = t_i + frame_offset`` (clamped ≥ 0),
  ``end = t_{i+1}`` (last cue ends at segment end).
- Windows shorter than ``min_cue_duration`` drop the short (earlier) cue's
  frame and keep the later change.
- Opening cue ``[0, t_1)`` (the title slide) only when ``t_1 > 5 s`` and an
  ``opening_description`` is supplied.
- Long descriptions split into consecutive cues whose durations are
  allocated proportionally to text length inside the slide window; cues
  never exceed the window. A split cue window longer than
  ``max_cue_duration_split`` gets enough cues to stay under that duration
  (a single short sentence — or an unbreakable long token such as a URL —
  may legitimately span the whole window).
- Text is wrapped at ``max_cue_chars`` via ``subtitles.wrap_text`` and the
  final list is passed through ``subtitles.normalize_cues`` (non-overlap,
  clamping, re-indexing).
"""

from __future__ import annotations

import logging
import math

from .subtitles import Cue, normalize_cues, wrap_text

log = logging.getLogger("vsa.slides")

#: The opening-slide cue is only emitted when the first change is later than this.
OPENING_MIN_T = 5.0


def build_slide_cues(
    changes: list[float],
    descriptions: list[str | None],
    segment_duration: float,
    *,
    frame_offset: float = 1.5,
    min_cue_duration: float = 1.0,
    opening_slide: bool = True,
    opening_description: str | None = None,
    max_cue_chars: int = 42,
    max_cue_duration_split: float = 30.0,
) -> list[Cue]:
    """Build the slide-track cues.

    ``changes``: accepted slide-change times (segment-relative seconds,
    unsorted OK); ``descriptions``: parallel to ``changes`` (index-aligned
    VLM texts, ``None`` for failed/unavailable frames).
    """
    pairs = sorted(
        ((float(t), d) for t, d in zip(changes, descriptions)), key=lambda p: p[0]
    )

    # Drop cues that survive the min-change interval yet leave a window
    # shorter than min_cue_duration: remove the short (earlier) cue, keep
    # the later one (its window then extends backwards to the survivor
    # before it).
    kept: list[tuple[float, str | None]] = []
    for t, desc in pairs:
        while kept and t - (kept[-1][0] + frame_offset) < min_cue_duration:
            dropped = kept.pop()
            log.debug("slide cues: dropping short cue at t=%.2fs", dropped[0])
        kept.append((t, desc))
    while kept and segment_duration - (kept[-1][0] + frame_offset) < min_cue_duration:
        dropped = kept.pop()
        log.debug("slide cues: dropping short final cue at t=%.2fs", dropped[0])

    cues: list[Cue] = []

    def emit(start: float, end: float, text: str) -> None:
        cues.append(Cue(index=len(cues) + 1, start=start, end=end, text=text))

    def emit_slide(start: float, end: float, text: str) -> None:
        """Split ``text`` proportionally across the ``[start, end)`` window."""
        dur = end - start
        if dur <= 0:
            return
        # Never request more parts than there are words to fill them; a part
        # count the splitter cannot honour would make the proportional split
        # drift (and the old refinement loop's precondition false).
        n = max(1, math.ceil(len(text) / max(1, max_cue_chars)))
        if n > 1 and dur > max_cue_duration_split:
            # enough text for several cues: keep each cue ≤ max_cue_duration_split
            n = max(n, math.ceil(dur / max_cue_duration_split))
        n = min(n, max(1, len(text.split())))
        if min_cue_duration > 0:
            n = max(1, min(n, int(dur / min_cue_duration)))
        parts = _split_text(text, n)
        max_parts = len(text.split())
        while len(parts) > 1 and len(parts) < max_parts:
            # grow the part count until no proportionally-long part exceeds
            # max_cue_duration_split; _split_text returns exactly min(words, n)
            # parts now, so this strictly progresses (the old loop could
            # stagnate on a fixed point and spin forever).
            total = sum(len(p) for p in parts)
            longest = max(len(p) for p in parts) / total * dur
            if longest <= max_cue_duration_split + 1e-9:
                break
            if min_cue_duration > 0 and dur / (len(parts) + 1) < min_cue_duration:
                break
            grow = math.ceil(len(parts) * longest / max_cue_duration_split)
            n = min(max(len(parts) + 1, grow), max_parts)
            parts = _split_text(text, n)
        total = sum(len(p) for p in parts) or 1
        pos = start
        for k, part in enumerate(parts):
            part_dur = dur * len(part) / total
            part_end = end if k == len(parts) - 1 else pos + part_dur
            emit(pos, part_end, wrap_text(part, max_cue_chars))
            pos = part_end

    # Opening slide (title slide before the first detected change).
    first_t = kept[0][0] if kept else segment_duration
    if opening_slide and first_t > OPENING_MIN_T:
        if opening_description:
            emit_slide(0.0, min(first_t, segment_duration), opening_description)
        else:
            log.debug("slide cues: opening slide at t=0 has no description; skipping cue")

    # One cue per surviving change: [t_i + frame_offset, t_{i+1}).
    for k, (t, desc) in enumerate(kept):
        start = min(max(0.0, t + frame_offset), segment_duration)
        end = kept[k + 1][0] if k + 1 < len(kept) else segment_duration
        end = min(end, segment_duration)
        if not desc:
            log.debug("slide cues: no description for slide at t=%.2fs; skipping cue", t)
            continue
        emit_slide(start, end, desc)

    cues = normalize_cues(cues, segment_duration=segment_duration)
    log.debug("slide cues: %d cue(s) for %d change(s)", len(cues), len(changes))
    return cues


def _split_text(text: str, n: int) -> list[str]:
    """Split ``text`` into exactly ``n`` parts at word boundaries, each
    ~``len(text)/n`` chars (parts stay in reading order, every part
    non-empty).

    With fewer than ``n`` words the text is split one word per part instead.
    The count is guaranteed: closes happen while the remaining word supply
    can still fill the parts still owed, and near the end each remaining
    word is forced into its own part.
    """
    if n <= 1:
        return [text]
    words = text.split()
    if len(words) <= n:
        return words
    target = max(1.0, len(text) / n)
    parts: list[list[str]] = []
    cur: list[str] = []
    size = 0
    for i, word in enumerate(words):
        cur.append(word)
        size += len(word) + 1
        remaining_words = len(words) - i - 1
        # parts still owed after the current one closes
        to_close = n - len(parts) - 1
        if (size >= target and remaining_words >= to_close) or remaining_words == to_close:
            parts.append(cur)
            cur, size = [], 0
    if cur:
        parts.append(cur)
    return [" ".join(p) for p in parts]
