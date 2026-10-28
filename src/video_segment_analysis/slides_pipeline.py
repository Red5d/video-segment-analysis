"""Process 3 — `vsa slides` orchestration (DESIGN §9).

Per segment: sample frames → detect changes (with min-change-interval
suppression) → extract full-res frames → describe them with the VLM →
write the slide-description subtitle track. Independent of transcription;
resumable through ``slides.status``.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rich.console import Console
from rich.table import Table

from . import ffmpegio
from .config import Config
from .errors import ManifestError, VSAError
from .manifest import Manifest, Segment, SlideChange, SlideFrame, stage_at_least
from .slide_cues import build_slide_cues
from .slidedetect import detect_changes
from .subtitles import write_srt, write_vtt

logger = logging.getLogger("vsa.slides")
console = Console()

OPENING_FRAME_TIME = 1.0   # s: probe time of the title slide (§9.5)
OPENING_KEY = -1.0         # sentinel `time` for the opening frame entry


def _slides_dir(workdir: Path, seg_id: str) -> Path:
    return workdir / "segments" / seg_id / "slides"


def run_slides(cfg: Config, *, segment_ids: list[str] | None = None,
               force: bool = False, dry_run: bool = False) -> int:
    workdir = cfg.paths.workdir
    manifest = Manifest.load(workdir)
    if not stage_at_least(manifest.pipeline.stage, "cut"):
        raise ManifestError(
            f"pipeline.stage={manifest.pipeline.stage!r}: run `vsa segment` "
            "first (needs stage >= cut)")

    targets = [s for s in manifest.select(segment_ids)
               if s.cut and s.cut.video and (force or s.slides.status != "done")]
    if not targets:
        logger.info("nothing to do for slides (all segments done)")
        return 0

    if dry_run:
        _print_dry_run(targets, cfg)
        return 0

    if force:
        for seg in targets:
            seg.slides = type(seg.slides)()  # fresh SlidesInfo

    # vlm client: lazy, built on first describe need
    failures: list[str] = []
    for seg in targets:
        video = workdir / seg.cut.video  # type: ignore[union-attr]
        if not video.is_file():
            logger.error("[slides:%s] video missing: %s", seg.id, video)
            seg.slides.status = "detect_failed"
            failures.append(seg.id)
            manifest.save(workdir)
            continue
        try:
            _process_segment(workdir, seg, video, cfg, _lazy_vlm(cfg))
        except VSAError as exc:
            logger.error("[slides:%s] failed: %s", seg.id, exc)
            if seg.slides.status in ("detecting", "pending",
                                     "slide_detect_suspect"):
                seg.slides.status = "detect_failed"
            else:
                seg.slides.status = "describe_failed"
            failures.append(seg.id)
        manifest.save(workdir)

    _print_summary(manifest)
    return 2 if failures else 0


def _lazy_vlm(cfg: Config):
    box: dict = {}

    def get():
        if "c" not in box:
            from .vlm import build_vlm_client
            box["c"] = build_vlm_client(cfg)
        return box["c"]
    return get


def _process_segment(workdir: Path, seg: Segment, video: Path, cfg: Config,
                     get_vlm) -> None:
    sl = seg.slides
    seg_dur = ffmpegio.probe(video).duration
    sdir = _slides_dir(workdir, seg.id)

    # ---- 1. sample + detect (§9.1/§9.2) -----------------------------------
    if not sl.changes:
        logger.info("[slides:%s] sampling frames at %.2f fps …", seg.id,
                    cfg.slides.sample_fps)
        frames = ffmpegio.sample_frames(video, cfg.slides)
        if frames.shape[0] < 2:
            raise ManifestError(f"[slides:{seg.id}] decoded {frames.shape[0]} frames")
        result = detect_changes(frames, cfg.slides)
        del frames  # free memory before the next segment
        sl.changes = [SlideChange(time=c.t, ratio=c.ratio) for c in result.changes]
        sl.suspect = result.suspect
        sl.status = "extracting"
        logger.info("[slides:%s] %d slide changes detected%s", seg.id,
                    len(sl.changes),
                    " — SUSPECT: max_frames_per_segment hit" if result.suspect else "")
    else:
        logger.info("[slides:%s] resuming: %d changes already detected",
                    seg.id, len(sl.changes))

    # ---- 2. extract frames (§9.3) ------------------------------------------
    existing = {f.time: f for f in sl.frames}
    kept: list[SlideFrame] = []
    for i, ch in enumerate(sl.changes, start=1):
        f = existing.pop(ch.time, None)
        if f is not None and (workdir / f.frame).is_file():
            kept.append(f)
            continue
        t = min(ch.time + cfg.slides.frame_offset, max(0.5, seg_dur - 0.5))
        dst = sdir / f"slide_{i:03d}.jpg"
        ffmpegio.extract_frame(video, t, dst, crop=cfg.slides.crop)
        kept.append(SlideFrame(time=ch.time, frame=str(dst.relative_to(workdir))))
    # opening (title) slide: only when the first change arrives late (§9.5)
    first_change = sl.changes[0].time if sl.changes else seg_dur
    opening = existing.pop(OPENING_KEY, None)
    if first_change > 5.0:
        if opening is not None and (workdir / opening.frame).is_file():
            kept.append(opening)
        else:
            dst = sdir / "slide_000.jpg"
            ffmpegio.extract_frame(video, min(OPENING_FRAME_TIME, seg_dur - 0.5),
                                   dst, crop=cfg.slides.crop)
            kept.append(SlideFrame(time=OPENING_KEY,
                                   frame=str(dst.relative_to(workdir))))
    sl.frames = kept
    sl.opening_frame = next((f.frame for f in kept if f.time == OPENING_KEY), None)
    sl.status = "describing"

    # ---- 3. describe frames (VLM, §9.4) -------------------------------------
    pending = [f for f in sl.frames if f.vlm_status == "pending"]
    if pending:
        logger.info("[slides:%s] describing %d frames (VLM backend %s)",
                    seg.id, len(pending), cfg.vlm.backend)
        vlm = get_vlm()
        _describe_frames(sl.frames, vlm, workdir, cfg)

    # ---- 4. slide subtitle track (§9.5) --------------------------------------
    change_frames = [f for f in sl.frames if f.time >= 0]
    times = [f.time for f in change_frames]
    descs = [f.description if f.vlm_status == "done" else None
             for f in change_frames]
    opening_f = next((f for f in sl.frames if f.time == OPENING_KEY), None)
    opening_desc = opening_f.description if (opening_f
                                             and opening_f.vlm_status == "done") else None
    cues = build_slide_cues(times, descs, seg_dur,
                            frame_offset=cfg.slides.frame_offset,
                            opening_description=opening_desc,
                            max_cue_chars=cfg.subtitles.max_cue_chars)
    if "srt" in cfg.subtitles.formats:
        write_srt(workdir / "segments" / f"{seg.id}.slides.srt", cues)
        sl.srt = f"segments/{seg.id}.slides.srt"
    if "vtt" in cfg.subtitles.formats:
        write_vtt(workdir / "segments" / f"{seg.id}.slides.vtt", cues)
        sl.vtt = f"segments/{seg.id}.slides.vtt"
    failed = [f for f in sl.frames if f.vlm_status == "vlm_failed"]
    sl.status = "done"
    logger.info("[slides:%s] done: %d slide cues → %s%s", seg.id, len(cues),
                ", ".join(x for x in (sl.srt, sl.vtt) if x),
                f" ({len(failed)} frames vlm_failed — repair with --segment-id)"
                if failed else "")


def _describe_frames(frames: list[SlideFrame], vlm, workdir: Path,
                     cfg: Config) -> None:
    """Describe pending frames concurrently; failures become placeholders."""
    from .vlm.base import SlideDescription

    def _one(f: SlideFrame) -> tuple[SlideFrame, SlideDescription]:
        path = workdir / f.frame
        try:
            return f, vlm.describe_checked(path)
        except VSAError as exc:  # client-side hard failure
            logger.warning("[slides] describe failed for %s: %s", f.frame, exc)
            return f, SlideDescription(text="[slide content unavailable]",
                                       failed=True)

    todo = [f for f in frames if f.vlm_status == "pending"]
    with ThreadPoolExecutor(max_workers=max(1, cfg.vlm.concurrency)) as pool:
        for frame, desc in pool.map(_one, todo):
            frame.vlm_status = "vlm_failed" if desc.failed else "done"
            frame.description = desc.text


def _print_dry_run(targets: list[Segment], cfg: Config) -> None:
    table = Table(title=f"slides plan ({len(targets)} segments, "
                       f"vlm backend {cfg.vlm.backend})")
    table.add_column("segment")
    table.add_column("video")
    table.add_column("known changes")
    table.add_column("max VLM calls")
    for seg in targets:
        table.add_row(seg.id, (seg.cut.video if seg.cut else "-") or "-",
                      str(len(seg.slides.changes)),
                      str(len(seg.slides.changes) + 1))
    console.print(table)
    console.print(f"detector: change_ratio={cfg.slides.change_ratio} "
                  f"pixel_diff={cfg.slides.pixel_diff_threshold} "
                  f"min_change_interval={cfg.slides.min_change_interval}s "
                  f"stable_frames={cfg.slides.stable_frames} "
                  f"sample_fps={cfg.slides.sample_fps} "
                  f"cap={cfg.slides.max_frames_per_segment}")


def _print_summary(manifest: Manifest) -> None:
    table = Table(title="vsa slides summary")
    table.add_column("segment")
    table.add_column("slides status")
    table.add_column("changes")
    table.add_column("frames")
    table.add_column("srt")
    for seg in manifest.segments:
        table.add_row(seg.id, seg.slides.status, str(len(seg.slides.changes)),
                      str(len(seg.slides.frames)), seg.slides.srt or "-")
    console.print(table)
