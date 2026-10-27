"""Pure interval algebra on the absolute source timeline.

All functions take/return ``Interval = (start, end)`` float tuples, are
non-mutating, and assume half-open semantics ``[start, end)``. Input lists
are normalized (sorted, clamped) internally so callers need not pre-sort.
"""

from __future__ import annotations

from typing import Iterable, Sequence

Interval = tuple[float, float]


def normalize(intervals: Iterable[Interval], *, total_duration: float | None = None,
              eps: float = 1e-9) -> list[Interval]:
    """Sort by start, clamp to ``[0, total_duration]``, drop empty intervals,
    and merge intervals that touch/overlap."""
    out: list[Interval] = []
    for s, e in intervals:
        s = max(0.0, float(s))
        if total_duration is not None:
            e = min(float(e), float(total_duration))
        else:
            e = float(e)
        if e - s > eps:
            out.append((s, e))
    out.sort()
    merged: list[Interval] = []
    for s, e in out:
        if merged and s <= merged[-1][1] + eps:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def total(intervals: Sequence[Interval]) -> float:
    """Summed duration of (disjoint) intervals."""
    return sum(e - s for s, e in intervals)


def merge(intervals: Sequence[Interval], max_gap: float) -> list[Interval]:
    """Merge intervals separated by a gap smaller than ``max_gap``."""
    merged: list[Interval] = []
    for s, e in normalize(intervals):
        if merged and s - merged[-1][1] < max_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def invert(intervals: Sequence[Interval], total_duration: float) -> list[Interval]:
    """Complement of ``intervals`` within ``[0, total_duration]``.

    ``invert(silences, dur)`` yields the speech intervals.
    """
    out: list[Interval] = []
    cursor = 0.0
    for s, e in normalize(intervals, total_duration=total_duration):
        if s - cursor > 1e-9:
            out.append((cursor, s))
        cursor = max(cursor, e)
    if total_duration - cursor > 1e-9:
        out.append((cursor, total_duration))
    return out


def drop_short(intervals: Sequence[Interval], min_dur: float) -> list[Interval]:
    """Discard intervals shorter than ``min_dur``."""
    return [(s, e) for s, e in normalize(intervals) if e - s >= min_dur]


def group_runs(intervals: Sequence[Interval], gap: float) -> list[list[Interval]]:
    """Split sorted intervals into runs; a gap ``>= gap`` starts a new run."""
    runs: list[list[Interval]] = []
    for s, e in normalize(intervals):
        if runs and s - runs[-1][-1][1] < gap:
            runs[-1].append((s, e))
        else:
            runs.append([(s, e)])
    return runs


def pad(intervals: Sequence[Interval], before: float, after: float,
        total_duration: float) -> list[Interval]:
    """Extend each interval ``before`` s backwards and ``after`` s forwards,
    clamped to ``[0, total_duration]``; overlapping results are merged."""
    padded = [(max(0.0, s - before), min(float(total_duration), e + after))
              for s, e in intervals]
    return normalize(padded, total_duration=total_duration)


def split_at(interval: Interval, points: Sequence[float]) -> list[Interval]:
    """Split one interval at the given interior points."""
    s, e = interval
    cuts = sorted(p for p in points if s < p < e)
    out: list[Interval] = []
    prev = s
    for p in cuts:
        out.append((prev, p))
        prev = p
    out.append((prev, e))
    return out


def split_even_cap(interval: Interval, cap: float,
                   candidate_points: Sequence[float] = ()) -> list[Interval]:
    """Split ``interval`` into pieces of at most ``cap`` seconds.

    Prefers the (sorted) ``candidate_points`` — e.g. silence gaps — for
    boundaries: a piece ends at the latest candidate point that keeps the
    current piece ``<= cap``. Falls back to exact ``cap``-spaced cuts when a
    piece would exceed the cap with no usable candidate.
    """
    s, e = interval
    if cap <= 0 or e - s <= cap:
        return [(s, e)]
    points = [p for p in sorted(candidate_points) if s < p < e]
    out: list[Interval] = []
    start = s
    i = 0
    while e - start > cap:
        target = start + cap
        # latest candidate that keeps the piece within the cap
        best: float | None = None
        while i < len(points) and points[i] <= target:
            if points[i] > start:
                best = points[i]
            i += 1
        if best is not None and best - start >= cap * 0.5:
            out.append((start, best))
            start = best
            # allow candidate reuse if it is the only way to keep pieces legal
            while i < len(points) and points[i] <= start:
                i += 1
        else:
            out.append((start, target))
            start = target
    out.append((start, e))
    return [(a, b) for a, b in out if b > a]
