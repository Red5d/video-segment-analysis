"""Chunk-audio cutting (transcribe §8.3): dst straight into atomic cut."""

from __future__ import annotations

from pathlib import Path

import pytest

from video_segment_analysis import ffmpegio
from video_segment_analysis.config import Config
from video_segment_analysis.errors import MediaError
from video_segment_analysis.manifest import Segment
from video_segment_analysis.asr.base import Utterance, Word
from video_segment_analysis.transcribe_pipeline import (
    _chunk_result_path, _cut_chunk_audio, _load_cached, _save_chunk)

CFG = Config()


def mk_segment() -> Segment:
    return Segment(id="seg_001", index=1, start=100.0, end=500.0,
                   duration=400.0)


@pytest.fixture
def workdir(tmp_path) -> Path:
    segdir = tmp_path / "segments"
    segdir.mkdir()
    (segdir / "seg_001.wav").write_bytes(b"RIFF....")
    return tmp_path


def test_cut_chunk_audio_passes_dst_directly(workdir, monkeypatch):
    seen = {}

    def fake_cut(src, dst, start, end, cfg):
        seen.update(src=src, dst=dst, start=start, end=end)
        # emulate ffmpegio._atomic_media writing and renaming into dst
        dst.write_bytes(b"RIFF....")

    monkeypatch.setattr(ffmpegio, "cut_audio_wav", fake_cut)
    dst = workdir / "segments" / "seg_001" / "asr" / "chunk_000.wav"
    _cut_chunk_audio(workdir, mk_segment(), (0.0, 123.4), dst, CFG)

    assert seen["dst"] == dst  # no stray .tmp suffix — muxer infers from .wav
    assert seen["src"] == workdir / "segments" / "seg_001.wav"
    assert seen["start"] == 0.0 and seen["end"] == pytest.approx(123.4)
    assert dst.is_file()


def test_cut_chunk_audio_skips_cached(workdir, monkeypatch):
    dst = workdir / "segments" / "seg_001" / "asr" / "chunk_000.wav"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"RIFF" + b"\0" * 100)  # > 44-byte header: keep
    monkeypatch.setattr(
        ffmpegio, "cut_audio_wav",
        lambda *a, **k: pytest.fail("must not re-cut a cached chunk"))
    _cut_chunk_audio(workdir, mk_segment(), (0.0, 100.0), dst, CFG)


def test_cut_chunk_audio_replaces_header_only_file(workdir, monkeypatch):
    dst = workdir / "segments" / "seg_001" / "asr" / "chunk_000.wav"
    dst.parent.mkdir(parents=True)
    dst.write_bytes(b"RIFF" + b"\0" * 20)  # ≤ 44 bytes: truncate artifact, re-cut
    monkeypatch.setattr(ffmpegio, "cut_audio_wav",
                        lambda s, d, a, b, c: d.write_bytes(b"RIFF...."))
    _cut_chunk_audio(workdir, mk_segment(), (0.0, 100.0), dst, CFG)
    assert dst.stat().st_size == 8


def test_cut_chunk_audio_requires_segment_wav(tmp_path, monkeypatch):
    monkeypatch.setattr(
        ffmpegio, "cut_audio_wav",
        lambda *a, **k: pytest.fail("no source → must fail before cutting"))
    from video_segment_analysis.errors import ManifestError
    with pytest.raises(ManifestError, match="vsa segment"):
        _cut_chunk_audio(tmp_path, mk_segment(), (0.0, 100.0),
                         tmp_path / "x" / "chunk_000.wav", CFG)


def test_atomic_media_rejects_suffixless_dst(tmp_path, monkeypatch):
    ran = []

    def fake_run(cmd, *, merge_stderr=False):
        ran.append(cmd)

    monkeypatch.setattr(ffmpegio, "run", fake_run)
    dst = tmp_path / "out_no_ext"
    with pytest.raises(MediaError, match="extension"):
        ffmpegio.cut_video_copy(tmp_path / "in.mp4", dst, 0, 10)
    assert not ran and not dst.exists()


# ---------------------------------------------------------------------------
# chunk result cache: JSON ⇄ Utterance/Word objects (nested words are dicts
# on disk — reconstructing them as objects is what cue assembly relies on)


def _sample_utterances():
    return [
        Utterance(0.0, 5.0, "hello world",
                  [Word("hello", 0.0, 1.2), Word("world", 1.3, 2.0)]),
        Utterance(5.0, 9.0, "no words here", None),
    ]


def test_chunk_cache_roundtrip(tmp_path):
    cdir = tmp_path / "seg_001" / "asr"
    utts = _sample_utterances()
    _save_chunk(cdir, 0, utts)
    got = _load_cached(cdir, 0)
    assert got == utts
    assert type(got[0].words[0]) is Word  # not a dict (the downstream crash)
    assert got[1].words is None


def test_chunk_cache_missing_corrupt(tmp_path):
    cdir = tmp_path / "seg_001" / "asr"
    assert _load_cached(cdir, 1) is None
    cdir.mkdir(parents=True)
    (_chunk_result_path(cdir, 1)).write_text("{not json", encoding="utf-8")
    assert _load_cached(cdir, 1) is None  # corrupt → treated as missing
