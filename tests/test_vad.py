"""VAD: silencedetect stderr parsing + detect_silences behaviour (§6.2)."""

from __future__ import annotations

import math

import pytest

from video_segment_analysis import ffmpegio, vad
from video_segment_analysis.config import SegmentationConfig, VadConfig
from video_segment_analysis.errors import SourceError

VAD = VadConfig(noise_db=-35, min_silence_duration=2.0)
SEG = SegmentationConfig(merge_speech_gap=2.0)

# Realistic ffmpeg stderr capture (interleaved noise, CRLF, trailing silence)
STDERR_TWO = (
    "ffmpeg version 9.0 Copyright (c) 2000-2026 the FFmpeg developers\r\n"
    "Input #0, wav, from 'audio.wav':\r\n"
    "  Duration: 00:00:20.00, bitrate: 256 kb/s\r\n"
    "Stream #0:0: Audio: pcm_s16le ([1][0][0][0] / 0x0001), 16000 Hz, mono, s16, 256 kb/s\r\n"
    "[silencedetect @ 0x55d1] silence_start: 3.2\r\n"
    "frame=    0 fps=0.0 q=0.0 size=N/A time=00:00:05.00 bitrate=N/A\r\n"
    "[silencedetect @ 0x55d1] silence_end: 8.7 | silence_duration: 5.5\r\n"
    "[silencedetect @ 0x55d1] silence_start: 12.05\r\n"
    "[silencedetect @ 0x55d1] silence_end: 15.55 | silence_duration: 3.5\r\n"
    "size=N/A time=00:00:20.00 bitrate=N/A speed= 120x\r\n"
)

STDERR_TRAILING = (
    "[silencedetect @ 0x1] silence_start: 10.0\n"
    "[silencedetect @ 0x1] silence_end: 14.0 | silence_duration: 4.0\n"
    "[silencedetect @ 0x1] silence_start: 18.0\n"
)

STDERR_DISCONTINUITY = (
    "[silencedetect @ 0x1] silence_start: 5.0\n"
    "[silencedetect @ 0x1] silence_start: 9.0\n"   # stream discontinuity
    "[silencedetect @ 0x1] silence_end: 12.0 | silence_duration: 3.0\n"
)

STDERR_UNMATCHED_END = (
    "[silencedetect @ 0x1] silence_end: 2.0 | silence_duration: 2.0\n"  # noise
    "[silencedetect @ 0x1] silence_start: 4.0\n"
    "[silencedetect @ 0x1] silence_end: 7.0 | silence_duration: 3.0\n"
)


def test_parse_two_silences():
    assert vad.parse_silencedetect(STDERR_TWO) == [(3.2, 8.7), (12.05, 15.55)]


def test_parse_open_ending_is_inf():
    out = vad.parse_silencedetect(STDERR_TRAILING)
    assert out[0] == (10.0, 14.0)
    assert out[1] == (18.0, math.inf)


def test_parse_discontinuity_closes_pending():
    # The pending (5.0, 9.0) is closed at the discontinuity; normalize()
    # then merges it with the touching (9.0, 12.0) into one interval.
    assert vad.parse_silencedetect(STDERR_DISCONTINUITY) == [(5.0, 12.0)]


def test_parse_ignores_unmatched_end_and_noise():
    assert vad.parse_silencedetect(STDERR_UNMATCHED_END) == [(4.0, 7.0)]


def test_parse_empty():
    assert vad.parse_silencedetect("no silencedetect output here\n") == []


def test_detect_silences_closes_trailing_and_drops_short(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....WAVEfmt ")

    def fake_run(cmd, *, merge_stderr=False):
        assert merge_stderr
        assert "silencedetect=noise=-35dB:d=2" in " ".join(cmd)
        return STDERR_TRAILING  # (10,14) kept; short/inf handled below

    monkeypatch.setattr(ffmpegio, "run", fake_run)
    sil = vad.detect_silences(wav, VadConfig(min_silence_duration=2.0), 20.0)
    assert sil == [(10.0, 14.0), (18.0, 20.0)]  # inf closed at duration


def test_detect_silences_drops_below_min_duration(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")

    def fake_run(cmd, *, merge_stderr=False):
        return ("[silencedetect @ 0x1] silence_start: 1.0\n"
                "[silencedetect @ 0x1] silence_end: 2.5 | silence_duration: 1.5\n"
                "[silencedetect @ 0x1] silence_start: 5.0\n"
                "[silencedetect @ 0x1] silence_end: 9.0 | silence_duration: 4.0\n")

    monkeypatch.setattr(ffmpegio, "run", fake_run)
    sil = vad.detect_silences(wav, VadConfig(min_silence_duration=2.0), 10.0)
    assert sil == [(5.0, 9.0)]


def test_detect_silences_zero_silence_raises_with_guidance(monkeypatch, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")

    def fake_run(cmd, *, merge_stderr=False):
        return "[silencedetect @ 0x1] size=N/A\n"

    monkeypatch.setattr(ffmpegio, "run", fake_run)
    with pytest.raises(SourceError, match="noise_db"):
        vad.detect_silences(wav, VAD, 10.0)


def test_silences_to_speech_inverts_and_merges():
    silences = [(0.0, 5.0), (10.0, 12.0), (30.0, 40.0)]
    speech = vad.silences_to_speech(silences, 50.0, VAD, SEG)
    # invert → (5,10),(12,30),(40,50); gap 12-10=2 NOT < merge gap 2 → no merge
    assert speech == [(5.0, 10.0), (12.0, 30.0), (40.0, 50.0)]
    merged = vad.silences_to_speech(silences, 50.0, VAD,
                                    SegmentationConfig(merge_speech_gap=2.5))
    assert merged == [(5.0, 30.0), (40.0, 50.0)]
