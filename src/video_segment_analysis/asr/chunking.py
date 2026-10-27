"""ASR chunk planning (DESIGN §8.3).

Speech intervals (segment-relative) are greedily merged into contiguous
chunks: a chunk boundary is always placed inside an inter-interval
silence gap, so chunks never truncate speech. Chunks tile the span from
the first speech interval to the last one (silence gaps are included
inside chunks); the target/cap knobs keep each alignment ≤ 5 min.
"""

from __future__ import annotations

from ..config import AsrConfig
from ..timeline import Interval, normalize


def plan_chunks(speech_intervals: list[Interval], segment_start: float,
                segment_end: float, cfg: AsrConfig) -> list[Interval]:
    """Plan utterance chunks for one segment's speech intervals.

    Greedy merge (§8.3): keep merging consecutive intervals while the
    *last merged interval ends within* ``max_chunk_seconds`` of the chunk
    start; the chunk boundary is then placed in the following silence gap
    (its midpoint, clamped so the chunk never exceeds the cap) — chunks
    never truncate speech. ``chunk_target_seconds`` is a soft preference
    only; the hard constraint is the cap (forced aligner: ≤ 5 min per
    alignment). A single speech interval longer than the cap becomes its
    own (oversized) chunk — it cannot be split without truncating speech.
    """
    ivs = [(max(s, segment_start), min(e, segment_end))
           for s, e in normalize(speech_intervals, total_duration=segment_end)
           if e > segment_start and s < segment_end]
    ivs = [(s, e) for s, e in ivs if e > s]
    if not ivs:
        return []

    cap = cfg.max_chunk_seconds
    chunks: list[Interval] = []
    n = len(ivs)
    i = 0
    cur = ivs[0][0]  # current chunk start; chunks tile contiguously
    while i < n:
        m = i  # last interval merged into the current chunk
        while m + 1 < n and ivs[m + 1][1] - cur <= cap:
            m += 1
        if m + 1 >= n:  # chunk runs to the last interval
            chunks.append((cur, ivs[m][1]))
            i = n
        else:
            gap_mid = (ivs[m][1] + ivs[m + 1][0]) / 2.0
            boundary = min(gap_mid, cur + cap)  # strictly inside the gap
            chunks.append((cur, boundary))
            cur = boundary  # next chunk starts inside the silence gap
            i = m + 1
    return chunks
