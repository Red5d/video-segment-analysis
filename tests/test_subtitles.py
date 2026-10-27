"""Subtitle writer/reader round-trips, normalization, ffmpeg parse smoke."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from video_segment_analysis.subtitles import (
    Cue, format_srt_time, format_vtt_time, normalize_cues, read_srt,
    read_vtt, wrap_text, write_srt, write_vtt,
)

# ---------------------------------------------------------------------------
# formatting


def test_format_times():
    assert format_srt_time(0) == "00:00:00,000"
    assert format_srt_time(3661.5) == "01:01:01,500"
    assert format_srt_time(-1) == "00:00:00,000"
    assert format_srt_time(99.9996) == "00:01:40,000"  # rounds up into minutes
    assert format_vtt_time(3661.5) == "01:01:01.500"
    assert format_vtt_time(1.2345) == "00:00:01.234"  # banker's rounding is fine


def test_wrap_text_keeps_words():
    out = wrap_text("the quick brown fox jumps over the lazy dog", 15)
    assert out.split("\n") == ["the quick brown", "fox jumps over", "the lazy dog"]
    assert wrap_text("supercalifragilistic x", 10) == "supercalifragilistic\nx"


# ---------------------------------------------------------------------------
# round-trips

CUES = [
    Cue(0.0, 2.5, "Hello there, welcome to the talk.", 1),
    Cue(3.0, 6.25, "Ünïcödé — 中文 ✓ — emojis 🎉 work.", 2),
    Cue(7.0, 12.0, "Multi line\ncue text", 3),
]


def _assert_same(a: list[Cue], b: list[Cue]) -> None:
    assert len(a) == len(b)
    for x, y in zip(a, b):
        assert x.index == y.index
        assert x.start == pytest.approx(y.start, abs=1e-3)
        assert x.end == pytest.approx(y.end, abs=1e-3)
        assert x.text == y.text


def test_srt_round_trip(tmp_path: Path):
    p = tmp_path / "out.srt"
    write_srt(p, CUES)
    _assert_same(read_srt(p), CUES)


def test_vtt_round_trip(tmp_path: Path):
    p = tmp_path / "out.vtt"
    write_vtt(p, CUES)
    raw = p.read_bytes()
    assert raw.startswith(b"WEBVTT\n\n")
    assert b"00:00:02.500" in raw
    _assert_same(read_vtt(p), CUES)


def test_srt_is_crlf_utf8_no_bom(tmp_path: Path):
    p = tmp_path / "out.srt"
    write_srt(p, CUES)
    raw = p.read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf")
    assert b"\r\n" in raw and b"\n" in raw
    assert "中文".encode("utf-8") in raw  # UTF-8, not escaped
    text = raw.decode("utf-8")
    assert text.count("\r\n\r\n") == 2  # blank line between cues


def test_read_tolerates_both_separators_and_hours(tmp_path: Path):
    srt = (
        "1\r\n0:01:30,500 --> 1:02:00,25\r\nComma style\r\n\r\n"
        "2\r\n00:02:00.000 --> 00:02:05.000\r\nDot style\r\nline2\r\n\r\n"
    )
    p = tmp_path / "mixed.srt"
    p.write_bytes(srt.encode("utf-8"))
    cues = read_srt(p)
    assert cues[0].start == pytest.approx(90.5)
    assert cues[0].end == pytest.approx(3720.25)
    assert cues[1].text == "Dot style\nline2"


def test_read_vtt_skips_header_note_and_cue_ids(tmp_path: Path):
    vtt = (
        "WEBVTT\n\n"
        "NOTE: comment block\n\n"
        "1\n00:00:01.000 --> 00:00:02.000 line:0\nFirst cue\n\n"
        "cue-id-2\n00:00:03.000 --> 00:00:04.000\nSecond cue\n\n"
    )
    p = tmp_path / "x.vtt"
    p.write_text(vtt, encoding="utf-8")
    cues = read_vtt(p)
    assert [c.text for c in cues] == ["First cue", "Second cue"]
    assert [c.index for c in cues] == [1, 2]


# ---------------------------------------------------------------------------
# normalization


def _norm(cues, **kw):
    return normalize_cues(cues, **kw)


def test_normalize_sorts_and_renumbers():
    cues = [Cue(5, 7, "second"), Cue(0, 2, "first")]
    out = _norm(cues)
    assert [c.text for c in out] == ["first", "second"]
    assert [c.index for c in out] == [1, 2]


def test_normalize_drops_empty_and_degenerate():
    cues = [Cue(0, 2, "   "), Cue(3, 3.05, "too short"), Cue(4, 3, "inverted"),
            Cue(5, 6, "kept")]
    out = _norm(cues)
    assert [c.text for c in out] == ["kept"]


def test_normalize_clamps_to_segment_duration():
    cues = [Cue(-1.0, 12.0, "wide")]
    out = _norm(cues, segment_duration=10.0)
    assert (out[0].start, out[0].end) == (0.0, 10.0)


def test_normalize_shrinks_overlap_and_drops_collapsed():
    cues = [
        Cue(0.0, 5.0, "overlaps next"),
        Cue(4.0, 6.0, "next cue"),
        Cue(6.5, 7.0, "collapsed by shrink: next starts right after"),
        Cue(7.2, 9.0, "final"),
    ]
    out = _norm(cues, min_gap=0.02, min_duration=0.2)
    for prev, nxt in zip(out, out[1:]):
        assert prev.end <= nxt.start - 0.02 + 1e-9
    assert "overlaps next" in [c.text for c in out]
    assert [c.index for c in out] == list(range(1, len(out) + 1))


def test_normalize_drops_cue_collapsed_by_shrink():
    out = _norm([Cue(0.0, 6.0, "early long"), Cue(0.1, 9.0, "overlapping")])
    assert [c.text for c in out] == ["overlapping"]


def test_normalize_wraps_text():
    out = _norm([Cue(0, 3, "a b c d e f g h i j k l m n o p")], max_width=10)
    assert "\n" in out[0].text
    assert all(len(line) <= 10 for line in out[0].text.split("\n"))


# ---------------------------------------------------------------------------
# ffmpeg smoke


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_ffmpeg_parses_outputs(tmp_path: Path):
    srt, vtt = tmp_path / "a.srt", tmp_path / "a.vtt"
    write_srt(srt, CUES)
    write_vtt(vtt, CUES)
    for fmt, p in (("srt", srt), ("webvtt", vtt)):
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", fmt,
             "-i", str(p), "-f", "webvtt", "-"],
            check=True, capture_output=True, timeout=60,
        )
