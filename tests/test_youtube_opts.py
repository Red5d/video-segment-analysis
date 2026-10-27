"""YouTube adapter: yt-dlp options, range strings, cut-strategy selection (§7.3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_segment_analysis import ffmpegio
from video_segment_analysis.config import Config
from video_segment_analysis.errors import SourceError
from video_segment_analysis.ffmpegio import MediaInfo
from video_segment_analysis.manifest import Segment
from video_segment_analysis.sources.youtube import YoutubeSource, section_range


def make_seg(start: float, end: float) -> Segment:
    return Segment(id="seg_001", index=1, start=start, end=end,
                   duration=end - start)


@pytest.fixture
def src(tmp_path) -> YoutubeSource:
    return YoutubeSource("https://youtu.be/x", Config(), tmp_path)


# ---------------------------------------------------------------------------
# section_range


@pytest.mark.parametrize("start,end,want", [
    (0, 59.6, "*0:00:00-0:01:00"),
    (3661.4, 7384, "*1:01:01-2:03:04"),
    (10.4, 20.6, "*0:00:10-0:00:21"),
    (-5, 30, "*0:00:00-0:00:30"),   # clamped at 0
])
def test_section_range(start, end, want):
    assert section_range(start, end) == want


# ---------------------------------------------------------------------------
# yt-dlp options


def test_base_opts_defaults(src):
    opts = src._base_opts()
    assert opts["noplaylist"] is True
    assert opts["retries"] == 3
    assert opts["fragment_retries"] == 10
    assert "cookiefile" not in opts


def test_base_opts_cookiefile(tmp_path):
    cfg = Config()
    cfg.source.cookiefile = "/secrets/cookies.txt"
    opts = YoutubeSource("u", cfg, tmp_path)._base_opts()
    assert opts["cookiefile"] == "/secrets/cookies.txt"


def test_ydl_merges_extra_opts(src, monkeypatch):
    captured: dict = {}

    class FakeYDL:
        def __init__(self, opts):
            captured.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setitem(__import__("sys").modules, "yt_dlp",
                        type("m", (), {"YoutubeDL": FakeYDL}))
    with src._ydl({"format": "best"}) as ydl:
        pass
    assert captured["format"] == "best"
    assert captured["noplaylist"] is True  # base opts preserved


# ---------------------------------------------------------------------------
# probe


def test_probe_requires_duration(src, monkeypatch):
    monkeypatch.setattr(src, "_extract_info", lambda: {"duration": 0})
    with pytest.raises(SourceError, match="live"):
        src.probe()


def test_probe_ok(src, monkeypatch):
    monkeypatch.setattr(src, "_extract_info",
                        lambda: {"duration": 120.5, "formats": [{}], "width": 1920})
    info = src.probe()
    assert info.duration == 120.5
    assert info.has_video and info.has_audio
    assert info.width == 1920


# ---------------------------------------------------------------------------
# cut_segment strategy selection + drift measurement


@pytest.fixture
def patched_media(tmp_path, monkeypatch):
    """Fake the heavy media ops; return a knob for the produced video duration."""
    full = tmp_path / "source_video.mp4"
    full.write_bytes(b"x")
    (tmp_path / "audio.wav").write_bytes(b"x")
    knob = {"out_dur": 60.0}

    monkeypatch.setattr(ffmpegio, "cut_video_copy",
                        lambda s, d, a, b: Path(d).write_bytes(b"v"))
    monkeypatch.setattr(ffmpegio, "cut_audio_wav",
                        lambda s, d, a, b, c: Path(d).write_bytes(b"w"))
    monkeypatch.setattr(ffmpegio, "probe",
                        lambda p: MediaInfo(duration=knob["out_dur"],
                                            has_video=True, has_audio=True))
    return knob


def test_cut_local_encode(src, patched_media, tmp_path, monkeypatch):
    encoded: list[tuple] = []
    monkeypatch.setattr(
        ffmpegio, "cut_video_encode",
        lambda s, d, a, b: (encoded.append((a, b)), Path(d).write_bytes(b"v")))
    src.full_video = tmp_path / "source_video.mp4"
    res = src.cut_segment(make_seg(100.0, 160.0), tmp_path / "v.mp4",
                          tmp_path / "a.wav", "local-encode")
    assert res.strategy == "local-encode"
    assert encoded == [(100.0, 160.0)]
    assert res.reencode_recommended is False
    assert res.actual_end - res.actual_start == pytest.approx(60.0)


def test_cut_local_copy_exact(src, patched_media, tmp_path):
    src.full_video = tmp_path / "source_video.mp4"
    res = src.cut_segment(make_seg(100.0, 160.0), tmp_path / "v.mp4",
                          tmp_path / "a.wav", "auto")
    assert res.strategy == "local-copy"
    assert res.actual_start == 100.0
    assert res.actual_end == pytest.approx(160.0)
    assert res.reencode_recommended is False
    assert res.video.is_file() and res.audio.is_file()


def test_cut_tail_overrun_flags_reencode(src, patched_media, tmp_path):
    src.full_video = tmp_path / "source_video.mp4"
    patched_media["out_dur"] = 65.0  # open-GOP copy ran 5s long
    res = src.cut_segment(make_seg(100.0, 160.0), tmp_path / "v.mp4",
                          tmp_path / "a.wav", "auto")
    assert res.reencode_recommended is True
    assert res.actual_start == 100.0
    assert res.actual_end == pytest.approx(165.0)


def test_cut_short_output_shifts_actual_start(src, patched_media, tmp_path):
    src.full_video = tmp_path / "source_video.mp4"
    patched_media["out_dur"] = 55.0  # 5s short: end trimmed at stream end
    res = src.cut_segment(make_seg(100.0, 160.0), tmp_path / "v.mp4",
                          tmp_path / "a.wav", "local-copy")
    assert res.actual_start == pytest.approx(95.0)
    assert res.reencode_recommended is True


def test_cut_yt_sections_falls_back_to_local_copy(src, patched_media, tmp_path):
    src.full_video = tmp_path / "source_video.mp4"
    res = src.cut_segment(make_seg(0.0, 60.0), tmp_path / "v.mp4",
                          tmp_path / "a.wav", "yt-sections")
    assert res.strategy == "local-copy"  # full video on disk → local cut is cheaper


def test_cut_without_full_video_requires_source_first(src, tmp_path):
    (tmp_path / "audio.wav").write_bytes(b"x")
    with pytest.raises(SourceError, match="full video"):
        src.cut_segment(make_seg(0.0, 60.0), tmp_path / "v.mp4",
                        tmp_path / "a.wav", "auto")


def test_cut_requires_full_wav(src, patched_media, tmp_path):
    src.full_video = tmp_path / "source_video.mp4"
    (tmp_path / "audio.wav").unlink()
    with pytest.raises(SourceError, match="audio.wav"):
        src.cut_segment(make_seg(0.0, 60.0), tmp_path / "v.mp4",
                        tmp_path / "a.wav", "auto")
