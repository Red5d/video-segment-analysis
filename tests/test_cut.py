"""Cutting: ffmpegio command construction + atomicity + real ffmpeg smoke (§7.3)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from video_segment_analysis import ffmpegio
from video_segment_analysis.config import AudioConfig
from video_segment_analysis.errors import MediaError

AUDIO = AudioConfig(sample_rate=16000, channels=1)


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Capture commands sent to ffmpegio.run; fake-create the output file."""
    calls: list[list[str]] = []

    def fake_run(cmd, *, merge_stderr=False):
        calls.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"fake-media")  # emulate ffmpeg writing output
        return ""

    monkeypatch.setattr(ffmpegio, "run", fake_run)
    return calls


def _val(cmd: list[str], flag: str) -> str:
    """Value token immediately following ``flag``."""
    return cmd[cmd.index(flag) + 1]


def test_cut_video_copy_command(captured, tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "seg.mp4"
    ffmpegio.cut_video_copy(src, dst, 12.5, 60.25)
    cmd = captured[0]
    assert cmd[0] == ffmpegio.FFMPEG
    # -ss/-to are *input* options: both before -i
    assert cmd.index("-ss") < cmd.index("-i") and cmd.index("-to") < cmd.index("-i")
    assert _val(cmd, "-ss") == "12.500"
    assert _val(cmd, "-to") == "60.250"  # absolute source-timeline end
    assert _val(cmd, "-c") == "copy"
    assert _val(cmd, "-avoid_negative_ts") == "make_zero"
    # written atomically: command targets the .part tmp file in dst's dir
    assert Path(cmd[-1]).parent == dst.parent and ".part" in Path(cmd[-1]).name
    assert dst.is_file()  # renamed by _atomic_media


def test_cut_video_encode_command(captured, tmp_path):
    src, dst = tmp_path / "in.mp4", tmp_path / "seg.mp4"
    ffmpegio.cut_video_encode(src, dst, 5.0, 65.0)
    cmd = captured[0]
    assert _val(cmd, "-c:v") == "libx264"
    assert _val(cmd, "-crf") == "22"
    assert _val(cmd, "-c:a") == "aac"
    assert _val(cmd, "-force_key_frames") == "expr:gte(t,n_forced*1)"


def test_cut_audio_wav_command(captured, tmp_path):
    src, dst = tmp_path / "full.wav", tmp_path / "seg.wav"
    ffmpegio.cut_audio_wav(src, dst, 100.0, 160.0, AUDIO)
    cmd = captured[0]
    assert cmd.index("-ss") < cmd.index("-i")  # input seek: byte-exact for PCM
    assert _val(cmd, "-ss") == "100.000000"
    assert _val(cmd, "-to") == "160.000000"
    assert _val(cmd, "-ar") == "16000"
    assert _val(cmd, "-ac") == "1"
    assert _val(cmd, "-c:a") == "pcm_s16le"


def test_atomic_media_writes_via_tmp_and_renames(captured, tmp_path):
    dst = tmp_path / "nested" / "out.mp4"
    ffmpegio.cut_video_copy(tmp_path / "in.mp4", dst, 0.0, 10.0)
    # command targets the .part tmp path, final rename happened
    assert captured[0][-1] != str(dst)
    assert ".part" in Path(captured[0][-1]).name
    assert dst.is_file()


def test_atomic_media_cleans_tmp_on_failure(monkeypatch, tmp_path):
    def boom(cmd, *, merge_stderr=False):
        Path(cmd[-1]).write_bytes(b"partial")
        raise MediaError("boom", cmd=cmd)

    monkeypatch.setattr(ffmpegio, "run", boom)
    dst = tmp_path / "out.mp4"
    with pytest.raises(MediaError, match="boom"):
        ffmpegio.cut_video_copy(tmp_path / "in.mp4", dst, 0.0, 10.0)
    assert not dst.exists()
    assert not any(p.name.startswith(".") for p in tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Real ffmpeg smoke (local binaries, no network)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_cut_copy_and_audio(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(
        [ffmpegio.FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=176x144:rate=25:duration=20",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
         "-c:v", "libx264", "-g", "50", "-c:a", "aac", str(src)],
        check=True, capture_output=True)

    out = tmp_path / "cut.mp4"
    ffmpegio.cut_video_copy(src, out, 5.0, 10.0)
    dur = ffmpegio.probe(out).duration
    assert 4.5 <= dur <= 11.0
    info = ffmpegio.probe(out)
    assert info.has_video and info.has_audio

    wav = tmp_path / "cut.wav"
    ffmpegio.cut_audio_wav(out, wav, 1.0, 3.0, AUDIO)
    info = ffmpegio.probe(wav)
    assert 1.9 <= info.duration <= 2.1  # -ss 1 -to 3 → 2 s
    assert info.has_audio and not info.has_video
    assert info.acodec == "pcm_s16le"
