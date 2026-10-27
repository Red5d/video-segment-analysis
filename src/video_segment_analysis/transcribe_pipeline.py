"""Process 2 — `vsa transcribe` orchestration (DESIGN §8).

Stage graph: load_manifest → plan_chunks → transcribe_chunks → build_cues
→ write_subtitles → finalize. Chunk completion is checkpointed (per-chunk
JSON + manifest counters), so a crash restarts only unfinished chunks.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.table import Table

from .asr.base import AsrClient, Utterance
from .asr.chunking import plan_chunks
from .config import Config
from .cues import utterances_to_cues
from .errors import ManifestError, VSAError
from .ffmpegio import probe
from .manifest import Manifest, Segment, stage_at_least
from .subtitles import write_srt, write_vtt
from .timeline import Interval

logger = logging.getLogger("vsa.transcribe")
console = Console()


def _chunk_dir(workdir: Path, seg_id: str) -> Path:
    return workdir / "segments" / seg_id / "asr"


def _chunk_audio_cached(cdir: Path, idx: int) -> Path:
    return cdir / f"chunk_{idx:03d}.wav"


def _chunk_result_path(cdir: Path, idx: int) -> Path:
    return cdir / f"chunk_{idx:03d}.json"


def _load_cached(cdir: Path, idx: int) -> list[Utterance] | None:
    p = _chunk_result_path(cdir, idx)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return [Utterance(**d) for d in data]
    except Exception:
        logger.warning("chunk result cache corrupt, re-transcribing: %s", p)
        p.unlink(missing_ok=True)
        return None


def _save_chunk(cdir: Path, idx: int, utts: list[Utterance]) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    p = _chunk_result_path(cdir, idx)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps([asdict(u) for u in utts], ensure_ascii=False),
                   encoding="utf-8")
    tmp.replace(p)


def run_transcribe(cfg: Config, *, segment_ids: list[str] | None = None,
                   force: bool = False, dry_run: bool = False) -> int:
    workdir = cfg.paths.workdir
    manifest = Manifest.load(workdir)
    if not stage_at_least(manifest.pipeline.stage, "cut"):
        raise ManifestError(
            f"pipeline.stage={manifest.pipeline.stage!r}: run `vsa segment` "
            "first (needs stage >= cut)")

    # segments ready for ASR; `transcribed` are skipped unless --force
    targets = manifest.select(segment_ids)
    targets = [s for s in targets
               if (s.status == "cut" or s.asr.status in
                   ("transcribing", "transcribe_failed"))
               and (force or s.asr.status != "transcribed")]
    if not targets:
        logger.info("nothing to transcribe (all segments transcribed)")
        return 0

    # ---- plan chunks --------------------------------------------------------
    plans: dict[str, list[Interval]] = {}
    for seg in targets:
        chunks_abs = plan_chunks(seg.speech_intervals, seg.start, seg.end, cfg.asr)
        plans[seg.id] = chunks_abs
        seg.asr.chunks_total = len(chunks_abs)
        seg.asr.backend = cfg.asr.backend
        seg.asr.model = cfg.asr.model if cfg.asr.backend == "qwen_local" else cfg.asr.openai_model
        seg.asr.aligner = cfg.asr.aligner
        seg.asr.chunk_plan = chunks_abs
    manifest.save(workdir)

    if dry_run:
        _print_dry_run(targets, plans, cfg.asr.backend)
        return 0

    client = _build_client(cfg)
    batchable = hasattr(client, "transcribe_batch")

    # ---- build job queue (chunk audio cut) ---------------------------------
    jobs: list[tuple[Segment, int, Interval]] = []
    failures: list[str] = []
    for seg in targets:
        cdir = _chunk_dir(workdir, seg.id)
        rel_plan = [(a - seg.start, b - seg.start) for a, b in plans[seg.id]]
        seg.asr.chunks_done = 0
        for idx, (a, b) in enumerate(rel_plan):
            if not force and _load_cached(cdir, idx) is not None:
                seg.asr.chunks_done += 1
                continue
            jobs.append((seg, idx, (a, b)))
    logger.info("transcribe: %d segments, %d chunks queued", len(targets), len(jobs))

    # ---- transcribe chunks ---------------------------------------------------
    def _run_job(seg: Segment, idx: int, rel: Interval) -> None:
        cdir = _chunk_dir(workdir, seg.id)
        wav = _chunk_audio_cached(cdir, idx)
        _cut_chunk_audio(workdir, seg, rel, wav, cfg)
        utts = client.transcribe(wav, cfg.asr.language)
        _save_chunk(cdir, idx, utts)

    def _run_batch(batch: list[tuple[Segment, int, Interval]]) -> None:
        wavs: list[Path] = []
        for seg, idx, rel in batch:
            cdir = _chunk_dir(workdir, seg.id)
            wav = _chunk_audio_cached(cdir, idx)
            _cut_chunk_audio(workdir, seg, rel, wav, cfg)
            wavs.append(wav)
        results = client.transcribe_batch(wavs, cfg.asr.language)  # type: ignore[attr-defined]
        for (seg, idx, _rel), utts in zip(batch, results):
            _save_chunk(_chunk_dir(workdir, seg.id), idx, utts)

    def _attempt(seg: Segment, idx: int, rel: Interval, fn) -> bool:
        prefix = f"transcribe: {seg.id} chunk {idx + 1}/{seg.asr.chunks_total}"
        try:
            fn()
            seg.asr.chunks_done += 1
            logger.info("%s ✓", prefix)
            return True
        except VSAError as exc:
            logger.error("%s failed: %s", prefix, exc)
            return False

    if batchable:
        # GPU-resident model: serialized batches of max_inference_batch_size.
        # A batch failure is re-attempted chunk-by-chunk to isolate the cause.
        batch_size = max(1, cfg.asr.max_inference_batch_size)
        for i in range(0, len(jobs), batch_size):
            batch = jobs[i:i + batch_size]
            first = batch[0]
            fn = (lambda b=batch: _run_batch(b)) if len(batch) > 1 \
                else (lambda j=first: _run_job(*j))
            if not _attempt(first[0], first[1], first[2], fn):
                if len(batch) == 1:
                    failures.append(first[0].id)
                else:
                    for seg, idx, rel in batch:
                        if not _attempt(seg, idx, rel,
                                        lambda s=seg, x=idx, r=rel: _run_job(s, x, r)):
                            failures.append(seg.id)
            manifest.save(workdir)
    else:
        def _one(job):
            seg, idx, rel = job
            ok = _attempt(seg, idx, rel, lambda: _run_job(seg, idx, rel))
            return seg.id if not ok else None
        if cfg.asr.concurrency > 1:
            with ThreadPoolExecutor(max_workers=cfg.asr.concurrency) as pool:
                failures += [x for x in pool.map(_one, jobs) if x]
        else:
            failures += [x for x in map(_one, jobs) if x]

    # ---- cues + subtitles ------------------------------------------------------
    for seg in targets:
        try:
            _write_subtitles(workdir, seg, plans[seg.id], cfg)
        except VSAError as exc:
            logger.error("[asr:%s] cue building/writing failed: %s", seg.id, exc)
            failures.append(seg.id)
            seg.asr.status = "transcribe_failed"
        manifest.save(workdir)

    _print_summary(manifest, targets)
    return 2 if failures else 0


def _build_client(cfg: Config) -> AsrClient:
    from .asr import build_asr_client
    return build_asr_client(cfg)


def _cut_chunk_audio(workdir: Path, seg: Segment, rel: Interval, dst: Path,
                     cfg: Config) -> None:
    from . import ffmpegio
    if dst.is_file() and dst.stat().st_size > 44:
        return
    src = workdir / "segments" / f"{seg.id}.wav"
    if not src.is_file():
        raise ManifestError(f"missing segment audio {src} — re-run `vsa segment`")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".wav.tmp")
    ffmpegio.cut_audio_wav(src, tmp, rel[0], rel[1], cfg.audio)
    tmp.replace(dst)


def _write_subtitles(workdir: Path, seg: Segment, plan: list[Interval],
                     cfg: Config) -> None:
    """Assemble cached chunk utterances → cues → SRT/VTT (§8.4/§8.5)."""
    cdir = _chunk_dir(workdir, seg.id)
    utterances: list[Utterance] = []
    missing = 0
    for idx, (abs_start, _abs_end) in enumerate(plan):
        utts = _load_cached(cdir, idx)
        if utts is None:
            missing += 1
            continue
        shift = abs_start - seg.start  # chunk start, segment-relative
        for u in utts:
            utterances.append(Utterance(
                start=u.start + shift, end=u.end + shift, text=u.text,
                words=[type(w)(w.text, w.start + shift, w.end + shift)
                       for w in u.words] if u.words else None))

    seg_dur = None
    if seg.cut and seg.cut.video:
        try:
            seg_dur = probe(workdir / seg.cut.video).duration
        except VSAError:
            seg_dur = None
    seg_dur = seg_dur or (seg.cut.actual_end - seg.cut.actual_start if seg.cut
                          else seg.duration)

    cues = utterances_to_cues(utterances, cfg=cfg.subtitles,
                              segment_duration=seg_dur)
    # Re-base onto the cut file: it starts `drift` seconds before seg.start
    # (keyframe snap, §8.5).
    drift = seg.cut_drift
    if drift:
        cues = [type(c)(max(0.0, c.start + drift), c.end + drift, c.text)
                for c in cues if c.end + drift > 0]

    partial = missing > 0
    if partial:
        logger.warning("[asr:%s] %d/%d chunks missing — writing partial SRT "
                       "(%s.srt.partial), segment marked transcribe_failed",
                       seg.id, missing, len(plan), seg.id)
    seg.asr.status = "transcribe_failed" if partial else "transcribed"

    fmts = cfg.subtitles.formats
    if "srt" in fmts:
        if partial:
            write_srt(workdir / "segments" / f"{seg.id}.srt.partial", cues)
        else:
            write_srt(workdir / "segments" / f"{seg.id}.srt", cues)
            seg.asr.srt = f"segments/{seg.id}.srt"
    if "vtt" in fmts and not partial:
        write_vtt(workdir / "segments" / f"{seg.id}.vtt", cues)
        seg.asr.vtt = f"segments/{seg.id}.vtt"
    if not partial:
        logger.info("[asr:%s] done: %d cues → %s", seg.id, len(cues),
                    ", ".join(x for x in (seg.asr.srt, seg.asr.vtt) if x))


def _print_dry_run(targets: list[Segment], plans: dict[str, list[Interval]],
                   backend: str) -> None:
    table = Table(title=f"transcribe plan ({len(targets)} segments, backend {backend})")
    table.add_column("segment")
    table.add_column("chunks")
    table.add_column("speech")
    table.add_column("max chunk")
    for seg in targets:
        ch = plans[seg.id]
        table.add_row(seg.id, str(len(ch)),
                      f"{sum(e - s for s, e in ch) / 60:.1f} min span",
                      f"{max((e - s for s, e in ch), default=0):.0f}s")
    console.print(table)


def _print_summary(manifest: Manifest, targets: list[Segment]) -> None:
    table = Table(title="vsa transcribe summary")
    table.add_column("segment")
    table.add_column("status")
    table.add_column("chunks")
    table.add_column("srt")
    table.add_column("vtt")
    for seg in manifest.segments:
        if seg not in targets:
            continue
        chunks = f"{seg.asr.chunks_done}/{seg.asr.chunks_total}"
        table.add_row(seg.id, seg.asr.status, chunks,
                      seg.asr.srt or "-", seg.asr.vtt or "-")
    console.print(table)
