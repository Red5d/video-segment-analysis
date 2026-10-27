"""Slide-change detector tests on synthetic frame sequences (DESIGN §13.1)."""

from __future__ import annotations

import numpy as np
import pytest

from video_segment_analysis.config import SlidesConfig
from video_segment_analysis.slidedetect import DetectionResult, detect_changes, frames_close


def seq(*runs: tuple[int, int]) -> np.ndarray:
    """Build (N, 32, 32) uint8 frames from (count, gray_value) runs."""
    return np.concatenate(
        [np.full((n, 32, 32), v, dtype=np.uint8) for n, v in runs]
    )


def times(result: DetectionResult) -> list[float]:
    return [c.t for c in result.changes]


CFG = SlidesConfig()  # fps 1.0, threshold 28, ratio 0.5, interval 3.0, stable 2


def test_instant_cut_accepted_at_correct_time():
    frames = seq((10, 0), (12, 200))
    result = detect_changes(frames, CFG)
    assert times(result) == [10.0]
    assert result.changes[0].ratio == pytest.approx(1.0)
    assert result.suspect is False


def test_fade_accepted_after_stabilization_at_run_start():
    # 5 changing samples (0→40→80→120→160→200) then stable: the candidate
    # is accepted only once the image settles, at the FIRST frame of the run.
    frames = seq((5, 0), (1, 40), (1, 80), (1, 120), (1, 160), (1, 200), (6, 200))
    result = detect_changes(frames, CFG)
    assert times(result) == [5.0]


def test_animation_shorter_than_min_change_interval_gives_one_change():
    # Slide change at t=10 accepted; a second animation starting at t=12
    # falls inside the 3 s suppression window → exactly one change.
    frames = seq((10, 0), (1, 200), (1, 200), (1, 120), (1, 120), (1, 120), (6, 120))
    result = detect_changes(frames, CFG)
    assert times(result) == [10.0]


def test_repeated_animations_one_change_per_suppression_window():
    # Two animation bursts (t=10 & t=12, then t=16 & t=18), each pair
    # spaced 2 s < min_change_interval 3 s: each burst yields exactly one
    # accepted change (the in-window animation is discarded).
    frames = np.zeros((24, 32, 32), dtype=np.uint8)
    frames[10:12] = 200
    frames[12:16] = 120
    frames[16:18] = 180
    frames[18:24] = 60
    result = detect_changes(frames, CFG)
    assert times(result) == [10.0, 16.0]


def test_long_animation_candidate_starts_at_window_expiry():
    # Change accepted at t=10; a sustained animation covers t=13..15
    # (window expired at 13.0) → new candidate starts at the expiry, t=13.
    frames = seq((10, 0), (3, 200), (1, 120), (1, 60), (1, 20), (6, 20))
    result = detect_changes(frames, CFG)
    assert times(result) == [10.0, 13.0]


def test_flicker_rejected():
    # Single-sample spike at t=5 recovering to the pre-change image: the
    # run never yields a stable image *distinct* from before the spike.
    frames = seq((5, 0), (1, 200), (8, 0))
    result = detect_changes(frames, CFG)
    assert times(result) == []


def test_cursor_blink_below_change_ratio_ignored():
    frames = np.zeros((14, 32, 32), dtype=np.uint8)
    frames[10:12, 0:2, 0:2] = 200  # 4/1024 px blink ≪ change_ratio
    result = detect_changes(frames, CFG)
    assert times(result) == []


def test_candidate_pending_at_eof_accepted():
    frames = seq((10, 0), (3, 200))  # run still active at the last sample
    result = detect_changes(frames, CFG)
    assert times(result) == [10.0]


def test_first_change_before_2s_skipped():
    frames = seq((1, 0), (1, 200), (8, 200))  # segment-start artifact at t=1
    result = detect_changes(frames, CFG)
    assert times(result) == []
    # ...and a later real change is still detected afterwards
    frames = seq((1, 0), (1, 200), (8, 200), (2, 0), (6, 0))
    result = detect_changes(frames, CFG)
    assert times(result) == [10.0]


def test_suspect_cap_keeps_strongest_sorted_by_time():
    cfg = SlidesConfig(max_frames_per_segment=3)
    frames = np.zeros((34, 32, 32), dtype=np.uint8)
    frames[10:14] = 200                    # full-screen: ratio 1.0
    frames[14:18] = 200
    frames[14:18, :, 20:32] = 0            # 12/32 cols revert: below ratio, not a change
    frames[18:22] = 200
    frames[22:26] = 200
    frames[22:26, :, 12:32] = 0            # 20/32 cols revert: ratio 0.625
    frames[26:30] = 200                    # full-screen: ratio 1.0
    # t=30: full-screen revert to black: ratio 1.0  → 4 changes vs cap 3:
    # strongest kept = t10 (1.0), t22 (0.625), t30 (1.0), sorted by time.
    result = detect_changes(frames, cfg)
    assert result.suspect is True
    assert times(result) == [10.0, 22.0, 30.0]
    assert [c.ratio for c in result.changes] == pytest.approx([1.0, 0.625, 1.0])


def test_frames_close():
    a = np.zeros((8, 8), dtype=np.uint8)
    b = a.copy()
    b[:3, :3] = 200  # 9/64 = 14% pixels differ strongly
    assert frames_close(a, b, tol=0.05) is False
    assert frames_close(a, b, tol=0.2) is True
    b2 = a.copy()
    b2[0, 0] = 10  # below pixel threshold
    assert frames_close(a, b2) is True


def test_short_input_no_change():
    assert detect_changes(np.zeros((1, 32, 32), dtype=np.uint8), CFG).changes == []
