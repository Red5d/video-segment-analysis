"""Slide subtitle track builder tests (DESIGN §9.5)."""

from __future__ import annotations

import pytest

from video_segment_analysis.slide_cues import OPENING_MIN_T, _split_text, build_slide_cues


def windows(cues) -> list[tuple[float, float]]:
    return [(c.start, c.end) for c in cues]


def test_cue_windows_tile_the_segment():
    changes = [10.0, 20.0, 30.0]
    descs = ["Alpha slide", "Beta slide", "Gamma slide"]
    cues = build_slide_cues(changes, descs, 40.0, frame_offset=1.5, opening_slide=False)
    assert windows(cues) == pytest.approx([(11.5, 20.0), (21.5, 30.0), (31.5, 40.0)])
    assert [c.text for c in cues] == descs
    assert [c.index for c in cues] == [1, 2, 3]
    starts = [c.start for c in cues]
    assert starts == sorted(starts)
    assert all(e > s for s, e in windows(cues))
    assert all(s2 >= e1 for (s1, e1), (s2, _) in zip(windows(cues), windows(cues)[1:]))


def test_opening_cue_only_when_first_change_late():
    changes, descs = [10.0], ["Slide one"]
    cues = build_slide_cues(
        changes, descs, 40.0, opening_description="Title slide: Conference 2026"
    )
    assert windows(cues) == pytest.approx([(0.0, 10.0), (11.5, 40.0)])
    assert cues[0].text == "Title slide: Conference 2026"

    # first change at/below the 5 s threshold → no opening cue
    cues = build_slide_cues(
        [OPENING_MIN_T], ["Slide"], 40.0, opening_description="Title"
    )
    assert windows(cues) == pytest.approx([(6.5, 40.0)])  # t_1 + frame_offset

    # opening description missing → no opening cue
    cues = build_slide_cues(changes, descs, 40.0, opening_description=None)
    assert len(cues) == 1
    assert cues[0].start == pytest.approx(11.5)

    # opening_slide disabled
    cues = build_slide_cues(
        changes, descs, 40.0, opening_slide=False, opening_description="Title"
    )
    assert len(cues) == 1


def test_min_duration_violation_drops_shorter_keeps_later():
    # Two changes 0.4 s apart: the earlier cue's window is negative → the
    # earlier frame is dropped, the later one survives.
    cues = build_slide_cues(
        [10.0, 10.4], ["Early slide", "Later slide"], 40.0,
        frame_offset=1.5, opening_slide=False,
    )
    assert len(cues) == 1
    assert cues[0].start == pytest.approx(11.9)
    assert cues[0].text == "Later slide"

    # Final change too close to segment end → dropped, previous extends.
    cues = build_slide_cues(
        [10.0, 39.0], ["Main slide", "Tail slide"], 40.0,
        frame_offset=1.5, opening_slide=False,
    )
    assert len(cues) == 1
    assert windows(cues) == pytest.approx([(11.5, 40.0)])
    assert cues[0].text == "Main slide"


def test_long_description_split_proportionally_within_window():
    desc = "alpha beta gamma delta epsilon zeta eta theta iota kappa"  # 50 chars
    cues = build_slide_cues(
        [10.0], [desc], 40.0, frame_offset=0.0, max_cue_chars=12,
        min_cue_duration=1.0, opening_slide=False,
    )
    assert len(cues) >= 2
    # never exceeds the slide window
    assert all(10.0 <= s < e <= 40.0 for s, e in windows(cues))
    # consecutive (min_gap apart after normalize_cues), monotonic,
    # durations proportional to part text length
    assert cues[0].start == pytest.approx(10.0)
    assert cues[-1].end == pytest.approx(40.0)
    assert all(a.end <= b.start < a.end + 0.05 for a, b in zip(cues, cues[1:]))
    lens = [len(c.text) for c in cues]
    total = sum(lens)
    for c, L in zip(cues, lens):
        assert (c.end - c.start) == pytest.approx(30.0 * L / total, abs=0.05)
    joined = " ".join(c.text for c in cues).split()
    assert joined == desc.split()  # text preserved in reading order


def test_long_cue_duration_split():
    # Long text on a 100 s slide: split so no cue exceeds max_cue_duration_split.
    desc = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    cues = build_slide_cues(
        [0.0], [desc], 100.0, frame_offset=0.0,
        max_cue_duration_split=30.0, opening_slide=False,
    )
    assert len(cues) >= 2
    assert all(e - s <= 30.0 + 1e-9 for s, e in windows(cues))
    assert cues[0].start == pytest.approx(0.0)
    assert cues[-1].end == pytest.approx(100.0)

    # A single short sentence legitimately spans the whole window.
    cues = build_slide_cues(
        [0.0], ["Short slide text"], 100.0, frame_offset=0.0,
        max_cue_duration_split=30.0, opening_slide=False,
    )
    assert len(cues) == 1


def test_clamps():
    # frame_offset pushing start past segment end → cue clamped/dropped.
    cues = build_slide_cues([39.5], ["Late"], 40.0, frame_offset=1.5,
                            opening_slide=False)
    assert cues == []  # 41.0 clamps to 40.0, zero-width window dropped

    # negative offset: start clamped to ≥ 0.
    cues = build_slide_cues([1.0], ["Slide"], 40.0, frame_offset=-5.0,
                            opening_slide=False)
    assert cues[0].start == pytest.approx(0.0)


def test_missing_description_skips_cue():
    cues = build_slide_cues(
        [10.0, 20.0], [None, "Second slide"], 40.0, opening_slide=False
    )
    assert len(cues) == 1
    assert cues[0].text == "Second slide"
    assert cues[0].start == pytest.approx(21.5)


def test_no_changes_opening_cue_covers_segment():
    cues = build_slide_cues([], [], 120.0, opening_description="Only a title slide")
    assert windows(cues) == pytest.approx([(0.0, 120.0)])


def test_split_text_honours_part_count():
    # exact part count when the text has more words than parts requested
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    assert len(_split_text(text, 1)) == 1
    assert _split_text(text, 1) == [text]
    for n in range(2, 13):  # 12 words available
        parts = _split_text(text, n)
        assert len(parts) == n
        assert all(parts)
        assert " ".join(parts).split() == text.split()  # reading order kept
    # fewer words than parts → one word per part
    assert _split_text(text, 20) == text.split()


def test_giant_tokens_terminate():
    # Slides full of unbreakable long tokens (URLs etc.) used to send the
    # proportional-split refinement loop into a fixed-point spin
    # (_split_text is not monotonic in n → infinite CPU, no output).
    text = " ".join(
        "x" * 200 if i % 3 == 0 else w
        for i, w in enumerate(
            ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
             "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi"] * 8
        )
    )
    assert len(text.split()) == 128
    opening = " ".join(
        "https://ex.tld/a/b/" + str(i) if i % 4 == 0 else f"word{i}"
        for i in range(60)
    )
    cues = build_slide_cues(
        [777.0], [text], 1968.0, frame_offset=1.5, opening_slide=False,
        opening_description=None,
    )
    assert len(cues) >= 2
    assert all(778.5 <= s < e <= 1968.0 for s, e in windows(cues))
    # with an opening description the same must hold (the original repro was
    # a long window + long opening + slide texts full of giant tokens)
    cues = build_slide_cues(
        [777.0], [text], 1968.0, frame_offset=1.5,
        opening_description=opening,
    )
    assert cues[0].start == pytest.approx(0.0)
    assert all(0.0 <= s < e <= 1968.0 for s, e in windows(cues))
    joined = " ".join(c.text for c in cues if c.start < 778.5).split()
    assert joined == opening.split()  # opening text preserved in order
