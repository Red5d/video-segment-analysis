"""Raw ``subprocess`` wrappers around ffmpeg/ffprobe (no Python ffmpeg package).

Every long-running call logs its start; every failure raises
:class:`~.errors.MediaError` carrying the *redacted* command and the tail
of the tool's stderr (§11: nonzero exit → last 40 stderr lines + secrets
redacted). URLs have their query strings (``?itok=...``, signatures, tokens)
stripped and the argument after auth-style flags is replaced with
``<redacted>``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import AudioConfig, SlidesConfig
from .errors import MediaError, SourceError

logger = logging.getLogger("vsa.ffmpeg")

FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"

# Flags whose *value* may carry credentials/signed tokens — never log them.
_SECRET_FLAGS = frozenset({
    "-headers", "-user-agent", "-user_agent", "-referer",
    "-cookie", "-cookies", "-location", "-post_data",
})
_URL_RE = re.compile(r"^https?://[^?\s]+(?:\?.*)?$")


@dataclass
class MediaInfo:
    """Media facts extracted from ffprobe JSON."""

    duration: float
    has_video: bool
    has_audio: bool
    width: int | None = None
    height: int | None = None
    vcodec: str | None = None
    acodec: str | None = None


# ---------------------------------------------------------------------------
# Command execution helpers


def _redact_url(token: str) -> str:
    base = token.split("?", 1)
    return base[0] + "/?<redacted>" if len(base) == 2 and _URL_RE.match(token) else token


def _redact(cmd: list[str]) -> list[str]:
    out: list[str] = []
    skip_next = False
    for tok in cmd:
        if skip_next:
            out.append("<redacted>")
            skip_next = False
            continue
        out.append(_redact_url(tok))
        skip_next = tok.lower() in _SECRET_FLAGS
    return out


def _tail(text: str | None, lines: int = 40) -> str:
    if not text:
        return ""
    return "\n".join(text.splitlines()[-lines:])


def run(cmd: list[str], *, merge_stderr: bool = False) -> str:
    """Run ``cmd``, capturing stdout/stderr.

    Returns stdout (plus stderr appended when ``merge_stderr=True``, used by
    filters that report on stderr such as ``silencedetect``). On nonzero exit
    raises :class:`MediaError` with the redacted command and the last 40
    stderr lines.
    """
    logger.debug("run: %s", " ".join(_redact(cmd)))
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0:
        raise MediaError(
            f"command failed (exit {proc.returncode})",
            cmd=_redact(cmd),
            stderr=_tail(proc.stderr),
        )
    if merge_stderr and proc.stderr:
        return proc.stdout + ("\n" if not proc.stdout.endswith("\n") and proc.stdout else "") + proc.stderr
    return proc.stdout


def check_ffmpeg() -> None:
    """Verify ``ffmpeg`` and ``ffprobe`` are on PATH; raise ``SourceError`` if not."""
    missing = [t for t in (FFMPEG, FFPROBE) if shutil.which(t) is None]
    if missing:
        raise SourceError(
            f"required tool(s) not found on PATH: {', '.join(missing)} — "
            "install ffmpeg (e.g. `apt install ffmpeg`, `brew install ffmpeg`) "
            "and ensure both ffmpeg and ffprobe are on PATH"
        )
    try:
        proc = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True,
                              errors="replace", check=False)
        first = (proc.stdout or proc.stderr).splitlines()[0] if (proc.stdout or proc.stderr) else "unknown"
    except OSError:  # pragma: no cover
        first = "unknown"
    logger.info("ffmpeg check ok: %s", first)


# ---------------------------------------------------------------------------
# Probing


def probe(path: Path) -> MediaInfo:
    """Probe media facts via ``ffprobe -print_format json``."""
    cmd = [FFPROBE, "-v", "error", "-print_format", "json",
           "-show_format", "-show_streams", str(path)]
    out = run(cmd)
    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        raise MediaError(f"ffprobe returned unparseable JSON for {path}",
                         cmd=_redact(cmd), stderr=_tail(out)) from exc

    duration = 0.0
    fmt_duration = (data.get("format") or {}).get("duration")
    if fmt_duration:
        duration = float(fmt_duration)

    has_video = has_audio = False
    width = height = None
    vcodec = acodec = None
    for s in data.get("streams", []):
        ctype = s.get("codec_type")
        stream_dur = s.get("duration")
        if ctype == "video" and not has_video:
            has_video = True
            vcodec = s.get("codec_name")
            width, height = s.get("width"), s.get("height")
            if not duration and stream_dur:
                duration = float(stream_dur)
        elif ctype == "audio" and not has_audio:
            has_audio = True
            acodec = s.get("codec_name")
            if not duration and stream_dur:
                duration = float(stream_dur)
    return MediaInfo(duration=duration, has_video=has_video, has_audio=has_audio,
                     width=width, height=height, vcodec=vcodec, acodec=acodec)


def _probe_duration(path: Path) -> float:
    return probe(path).duration


# ---------------------------------------------------------------------------
# Audio


def _atomic_media(cmd: list[str], dst: Path) -> None:
    """Run an ffmpeg media-writing command atomically: tmp file + rename."""
    if not dst.suffix:
        raise MediaError(
            f"output path has no extension for ffmpeg to infer its format "
            f"from: {dst} — outputs must end in a standard extension "
            f"(e.g. .wav, .mp4)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.stem}.part{dst.suffix}")  # keep suffix: ffmpeg muxer by extension
    cmd = [*cmd[:-1], str(tmp)]
    try:
        run(cmd)
        os.replace(tmp, dst)
    except MediaError:
        tmp.unlink(missing_ok=True)
        raise


def extract_audio(src: Path, dst_wav: Path, cfg: AudioConfig) -> None:
    """Decode ``src`` to a 16 kHz mono ``pcm_s16le`` WAV (DESIGN §6.1), atomic."""
    logger.info("extract-audio: %s -> %s", src, dst_wav.name)
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-y", "-i", str(src),
           "-vn", "-ac", str(cfg.channels), "-ar", str(cfg.sample_rate),
           "-c:a", "pcm_s16le", str(dst_wav)]
    _atomic_media(cmd, dst_wav)


def cut_audio_wav(src_wav: Path, dst_wav: Path, start: float, end: float,
                  cfg: AudioConfig) -> None:
    """Sample-accurate cut ``[start, end)`` from a WAV (any ffmpeg-decodable
    audio source works). ``-ss``/``-to`` are input options: for PCM the input
    seek is byte-exact, so the result is sample-accurate and cheap (§7.3)."""
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-y",
           "-ss", f"{start:.6f}", "-to", f"{end:.6f}", "-i", str(src_wav),
           "-ac", str(cfg.channels), "-ar", str(cfg.sample_rate),
           "-c:a", "pcm_s16le", str(dst_wav)]
    _atomic_media(cmd, dst_wav)


# ---------------------------------------------------------------------------
# Video cutting


def cut_video_copy(src: Path, dst: Path, start: float, end: float) -> None:
    """Fast stream-copy cut: ``-ss`` before ``-i`` (fast seek), ``-to`` as an
    input option is the *absolute* source-timeline end. The start snaps to
    the previous keyframe (§7.3 ``local-copy``)."""
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-y",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
           "-c", "copy", "-avoid_negative_ts", "make_zero", str(dst)]
    _atomic_media(cmd, dst)


def cut_video_encode(src: Path, dst: Path, start: float, end: float) -> None:
    """Frame-accurate re-encode cut (§7.3 ``local-encode``): same fast seek
    plus libx264/aac with a keyframe forced at the start."""
    cmd = [FFMPEG, "-hide_banner", "-nostats", "-y",
           "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(src),
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
           "-c:a", "aac", "-force_key_frames", "expr:gte(t,n_forced*1)",
           str(dst)]
    _atomic_media(cmd, dst)


# ---------------------------------------------------------------------------
# Frame sampling / extraction


def _scale_height(src_w: int, src_h: int, target_w: int) -> int:
    """Height produced by ``scale=<target_w>:-2`` (aspect-preserving, even)."""
    h = int(round(src_h * target_w / src_w))
    if h % 2:
        h += 1  # -2 rounds to the nearest even value
    return max(2, h)


def sample_frames(video: Path, cfg: SlidesConfig) -> np.ndarray:
    """Stream-decode ``video`` at ``cfg.sample_fps`` into a ``(N, h, w)``
    uint8 grayscale array (DESIGN §9.1). Frame ``i`` corresponds to
    timestamp ``i / sample_fps`` (media-relative).

    Memory: ~1 h at 1 fps ≈ 3.6k frames × ``320×180`` bytes ≈ 200 MB;
    segments are processed sequentially. Never decodes at full resolution.
    """
    info = probe(video)
    if not info.has_video or not info.width or not info.height:
        raise MediaError(f"no video stream to sample: {video}")

    vf_parts = [f"fps={cfg.sample_fps:g}"]
    if cfg.crop_tuple:
        x, y, w, h = cfg.crop_tuple
        vf_parts.append(f"crop={x}:{y}:{w}:{h}")
    vf_parts += [f"scale={cfg.analyze_width}:-2", "format=gray"]

    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(video),
           "-vf", ",".join(vf_parts), "-pix_fmt", "gray", "-f", "rawvideo", "-"]
    logger.info("sample-frames: %s @ %g fps -> %dpx wide gray",
                video.name, cfg.sample_fps, cfg.analyze_width)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunks: list[bytes] = []
    assert proc.stdout is not None
    while True:
        block = proc.stdout.read(1 << 20)
        if not block:
            break
        chunks.append(block)
    stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    rc = proc.wait()
    if rc != 0:
        raise MediaError("frame sampling failed", cmd=_redact(cmd), stderr=_tail(stderr))

    data = np.frombuffer(b"".join(chunks), dtype=np.uint8)
    frame_bytes = cfg.analyze_width * _scale_height(info.width, info.height,
                                                    cfg.analyze_width)
    if data.size % frame_bytes:
        # Probe-derived height disagreed with ffmpeg's -2 rounding: try h±2.
        for delta in (-2, 2):
            cand = cfg.analyze_width * (_scale_height(info.width, info.height,
                                                      cfg.analyze_width) + delta)
            if cand > 0 and data.size % cand == 0:
                frame_bytes = cand
                break
        else:
            raise MediaError(
                f"rawvideo stream ({data.size} bytes) is not a multiple of the "
                f"frame size {frame_bytes} for {video}", cmd=_redact(cmd))
    h = frame_bytes // cfg.analyze_width
    return data[: data.size - data.size % frame_bytes].reshape(-1, h, cfg.analyze_width)


def extract_frame(video: Path, t: float, dst_jpg: Path,
                  crop: tuple[int, int, int, int] | None = None) -> None:
    """Extract one full-resolution JPEG at ~``t`` via fast seek (§9.3);
    ``crop=(x, y, w, h)`` optionally masks overlays (logos, lower thirds)."""
    dst_jpg.parent.mkdir(parents=True, exist_ok=True)
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
           "-ss", f"{max(0.0, t):.3f}", "-i", str(video)]
    if crop:
        x, y, w, h = crop
        cmd += ["-vf", f"crop={x}:{y}:{w}:{h}"]
    cmd += ["-frames:v", "1", "-q:v", "2", str(dst_jpg)]
    run(cmd)
