"""Process 1 — `vsa segment` orchestration (DESIGN §7).

Stage graph: resolve_source → extract_audio → detect_speech → segment →
cut_media → finalize. Each stage persists to the manifest; re-running
resumes from the first incomplete stage. No LLM/ASR calls here.
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import ffmpegio, timeline, vad
from .config import Config
from .errors import MediaError, SourceError, VSAError
from .manifest import CutInfo, Manifest, Segment, stage_at_least
from .sources import SourceAdapter, resolve_source

logger = logging.getLogger("vsa.segment")
console = Console()


def _fmt_hms(t: float) -> str:
    t = int(round(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{s:02d}"


def _valid_wav(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 100_000:
        return False
    try:
        ffmpegio.probe(path)
        return True
    except MediaError:
        return False


def plan_segments(speech: list[timeline.Interval], total_duration: float,
                  cfg: Config) -> list[tuple[float, float, list[timeline.Interval]]]:
    """Pure segmentation: drop short runs → group by min_silence_gap → pad.

    Returns (start, end, speech_intervals) per segment on the absolute
    source timeline (padded boundaries; speech intervals un-padded).
    """
    seg_cfg = cfg.segmentation
    speech = timeline.drop_short(speech, seg_cfg.min_segment_duration)
    runs = timeline.group_runs(speech, seg_cfg.min_silence_gap)
    out = []
    for run in runs:
        padded = timeline.pad(run, seg_cfg.pad_start, seg_cfg.pad_end, total_duration)
        out.append((padded[0][0], padded[-1][1], [tuple(iv) for iv in run]))
    return out


def run_segment(cfg: Config, source: str, *, strategy: str | None = None,
                segment_ids: list[str] | None = None,
                force: bool = False, dry_run: bool = False) -> int:
    """Entry point for `vsa segment`. Returns a process exit code."""
    workdir = cfg.paths.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    # resume any prior run
    try:
        manifest = Manifest.load(workdir)
        if (manifest.pipeline.config_hash
                and manifest.pipeline.config_hash != cfg.config_hash()):
            logger.warning("config hash differs from the one this manifest was "
                           "built with; results may be stale for changed tuning "
                           "parameters (use --force to redo)")
    except VSAError:
        manifest = Manifest(pipeline={"config_hash": cfg.config_hash()})
    manifest.pipeline.config_hash = cfg.config_hash()

    # ---- stage 1: resolve source ------------------------------------------
    adapter: SourceAdapter = resolve_source(source, cfg, workdir)
    info = adapter.probe()
    src_desc = adapter.describe_source()
    manifest.source.type = src_desc.get("type", manifest.source.type)
    manifest.source.url = src_desc.get("url")
    manifest.source.path = src_desc.get("path")
    manifest.source.title = src_desc.get("title")
    manifest.source.duration = info.duration
    logger.info("source: %s (%s, duration %s)",
                src_desc.get("title") or src_desc.get("path") or source,
                manifest.source.type, _fmt_hms(info.duration))
    manifest.save(workdir)

    audio_wav = workdir / "audio.wav"
    # --force with --segment-id targets the cut stage only (repair runs):
    # upstream stages are reused when their artifacts are present.
    force_full = force and not segment_ids

    # ---- stage 2: extract full audio --------------------------------------
    if not (stage_at_least(manifest.pipeline.stage, "audio_ready")
            and _valid_wav(audio_wav) and not force_full):
        if audio_wav.is_file():
            logger.info("re-extracting audio (existing %s is incomplete/corrupt)", audio_wav)
        logger.info("[extract_audio] decoding full audio → %s", audio_wav)
        adapter.extract_full_audio(audio_wav)
        audio_info = ffmpegio.probe(audio_wav)
        drift = abs(audio_info.duration - info.duration)
        manifest.source.drift_warning = drift if drift > 2.0 else None
        if drift > 2.0:
            logger.warning("audio duration deviates from source by %.1fs "
                           "(recorded as drift_warning)", drift)
        manifest.set_stage("audio_ready")
        manifest.save(workdir)
    else:
        logger.info("[extract_audio] audio.wav already present — skipping")

    audio_dur = ffmpegio.probe(audio_wav).duration

    # ---- stage 3: detect speech -------------------------------------------
    if not stage_at_least(manifest.pipeline.stage, "speech_detected") or force_full:
        logger.info("[detect_speech] silence detection (noise_db=%d, d=%.1fs)",
                    cfg.vad.noise_db, cfg.vad.min_silence_duration)
        silences = vad.detect_silences(audio_wav, cfg.vad, audio_dur)
        speech = vad.silences_to_speech(silences, audio_dur, cfg.vad, cfg.segmentation)
        manifest.vad.engine = cfg.vad.engine
        manifest.vad.noise_db = cfg.vad.noise_db
        manifest.vad.speech_intervals = speech
        manifest.set_stage("speech_detected")
        manifest.save(workdir)
        logger.info("[detect_speech] %d speech intervals, %s of speech",
                    len(speech), _fmt_hms(timeline.total(speech)))
    speech = [tuple(iv) for iv in manifest.vad.speech_intervals]

    # ---- stage 4: segment ---------------------------------------------------
    planned = plan_segments(speech, audio_dur, cfg)
    if dry_run:
        _print_dry_run(planned, cfg, audio_dur, strategy, info.has_video,
                       manifest.source.type)
        return 0

    if not stage_at_least(manifest.pipeline.stage, "segmented") or force_full:
        old_by_id = {s.id: s for s in manifest.segments}
        segments: list[Segment] = []
        for i, (start, end, intervals) in enumerate(planned, start=1):
            seg_id = f"seg_{i:03d}"
            prev = old_by_id.get(seg_id)
            same = (prev is not None and not force_full
                    and abs(prev.start - start) < 1.0 and abs(prev.end - end) < 1.0
                    and prev.status in ("cut", "transcribing", "transcribed"))
            seg = prev if same else Segment(id=seg_id, index=i, start=start, end=end,
                                            duration=end - start, status="pending")
            seg.speech_intervals = intervals
            seg.duration = end - start
            if not same and prev is not None and prev.status != "pending":
                logger.warning("%s boundaries changed; dependent stages reset", seg_id)
                seg.asr = type(seg.asr)()
                seg.slides = type(seg.slides)()
                seg.status = "pending"
            segments.append(seg)
        manifest.segments = segments
        manifest.set_stage("segmented")
        manifest.save(workdir)
        logger.info("[segment] %d segments planned", len(segments))
    else:
        segments = manifest.segments

    # ---- stage 5: cut media -------------------------------------------------
    strat = strategy or cfg.cut.strategy
    failures = _cut_media(cfg, manifest, adapter, segments, strat,
                          segment_ids=segment_ids, force=force)

    # ---- finalize -------------------------------------------------------------
    manifest.set_stage("cut")
    manifest.save(workdir)
    _print_summary(manifest)
    return 2 if failures else 0


def _resolve_cut_source(cfg: Config, manifest: Manifest, adapter: SourceAdapter,
                        strat: str, workdir: Path) -> tuple[str, SourceAdapter]:
    """For YouTube + auto: download the full video once, then cut locally."""
    from .ffmpegio import probe
    if manifest.source.type != "youtube":
        return ("local-copy" if strat == "auto" else strat), adapter
    if strat == "yt-sections":
        return "yt-sections", adapter
    full = workdir / "source_video.mp4"
    yt = adapter
    if not (full.is_file() and probe(full).duration > 0):
        logger.info("[cut] downloading full video once → %s", full)
        yt.download_full_video(full, cfg.source.video_format)  # type: ignore[attr-defined]
        manifest.source.downloaded_full = True
        manifest.save(workdir)
    yt.full_video = full  # type: ignore[attr-defined]
    return ("local-copy" if strat == "auto" else strat), yt


def _cut_media(cfg: Config, manifest: Manifest, adapter: SourceAdapter,
               segments: list[Segment], strat: str, *,
               segment_ids: list[str] | None, force: bool) -> list[str]:
    workdir = cfg.paths.workdir
    seg_dir = workdir / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    strat, adapter = _resolve_cut_source(cfg, manifest, adapter, strat, workdir)

    targets = manifest.select(segment_ids)
    failures: list[str] = []
    for seg in targets:
        if seg.id not in {s.id for s in segments}:
            continue
        dst_video = seg_dir / f"{seg.id}.mp4"
        dst_audio = seg_dir / f"{seg.id}.wav"
        done = (seg.status in ("cut", "transcribing", "transcribed")
                and dst_video.is_file() and _valid_wav(dst_audio))
        if done and not force:
            logger.info("[cut:%s] outputs exist — skipping", seg.id)
            continue
        logger.info("[cut:%s] %s → %s (%s)", seg.id, strat,
                    _fmt_hms(seg.start), _fmt_hms(seg.end))
        try:
            result = adapter.cut_segment(seg, dst_video, dst_audio, strat)
        except (MediaError, SourceError) as exc:
            logger.error("[cut:%s] failed: %s", seg.id, exc)
            seg.status = "cut_failed"
            manifest.save(workdir)
            failures.append(seg.id)
            continue
        seg.cut = CutInfo(
            strategy=result.strategy,
            requested_start=seg.start, requested_end=seg.end,
            actual_start=result.actual_start, actual_end=result.actual_end,
            video=str(Path(result.video).relative_to(workdir)) if result.video else None,
            audio=str(Path(result.audio).relative_to(workdir)) if result.audio else None,
            reencode_recommended=result.reencode_recommended,
        )
        if result.reencode_recommended:
            logger.warning(
                "[cut:%s] stream-copy duration deviates >3s (sparse keyframes); "
                "re-cut with --strategy local-encode --segment-id %s --force",
                seg.id, seg.id)
        seg.status = "cut"
        manifest.save(workdir)
    return failures


def _print_dry_run(planned, cfg: Config, total_duration: float,
                   strategy: str | None, has_video: bool, source_type: str) -> None:
    table = Table(title=f"segment plan ({len(planned)} segments, "
                       f"source { _fmt_hms(total_duration)})")
    table.add_column("id")
    table.add_column("start")
    table.add_column("end")
    table.add_column("duration")
    table.add_column("speech runs")
    for i, (start, end, intervals) in enumerate(planned, start=1):
        table.add_row(f"seg_{i:03d}", _fmt_hms(start), _fmt_hms(end),
                      _fmt_hms(end - start), str(len(intervals)))
    console.print(table)
    strat = strategy or cfg.cut.strategy
    if strat == "auto":
        strat = {"local": "local-copy", "youtube": "full-download + local-copy"}[source_type]
    console.print(f"cut strategy: [bold]{strat}[/bold]"
                  + ("" if has_video else "  [yellow](source has no video track)[/yellow]"))
    console.print(f"vad: noise_db={cfg.vad.noise_db} min_silence={cfg.vad.min_silence_duration} | "
                  f"segmentation: gap≥{cfg.segmentation.min_silence_gap}s splits, "
                  f"min {cfg.segmentation.min_segment_duration}s, pads "
                  f"{cfg.segmentation.pad_start}/{cfg.segmentation.pad_end}s")


def _print_summary(manifest: Manifest) -> None:
    table = Table(title="vsa segment summary")
    table.add_column("segment")
    table.add_column("span (source)")
    table.add_column("status")
    table.add_column("video")
    table.add_column("audio")
    for seg in manifest.segments:
        video = seg.cut.video if seg.cut and seg.cut.video else "-"
        audio = seg.cut.audio if seg.cut and seg.cut.audio else "-"
        table.add_row(seg.id, f"{_fmt_hms(seg.start)}–{_fmt_hms(seg.end)}",
                      seg.status, video, audio)
    console.print(table)
