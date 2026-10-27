"""manifest.json — canonical state of a run.

Written atomically (temp file + rename). Each process reads the whole
manifest and updates only its own fields; per-segment ``status`` (and
``asr.status`` / ``slides.status``) is the handoff between processes.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .errors import ManifestError

# ---------------------------------------------------------------------------
# Pipeline stages (ordinal progression)

STAGES = ["initialized", "audio_ready", "speech_detected", "segmented",
          "cut", "transcribed", "done"]


def stage_rank(stage: str) -> int:
    try:
        return STAGES.index(stage)
    except ValueError:
        return -1


def stage_at_least(stage: str, minimum: str) -> bool:
    return stage_rank(stage) >= stage_rank(minimum)


Stage = Literal["initialized", "audio_ready", "speech_detected", "segmented",
                "cut", "transcribed", "done"]

SegmentStatus = Literal["pending", "cut", "transcribing", "transcribed",
                        "source_failed", "audio_failed", "cut_failed",
                        "transcribe_failed"]
AsrStatus = Literal["pending", "transcribing", "transcribed", "transcribe_failed"]
SlidesStatus = Literal["pending", "detecting", "extracting", "describing",
                       "done", "detect_failed", "describe_failed",
                       "slide_detect_suspect"]


# ---------------------------------------------------------------------------
# Sub-models


class SourceInfo(BaseModel):
    type: Literal["local", "youtube"] = "local"
    url: str | None = None            # URL for youtube sources
    path: str | None = None           # local path (original or downloaded)
    title: str | None = None
    duration: float | None = None
    downloaded_full: bool = False
    drift_warning: float | None = None  # |audio_dur - source_dur| in seconds


class PipelineInfo(BaseModel):
    stage: Stage = "initialized"
    config_hash: str = ""


class VadInfo(BaseModel):
    engine: str = "ffmpeg"
    noise_db: int = -35
    speech_intervals: list[tuple[float, float]] = Field(default_factory=list)


class CutInfo(BaseModel):
    strategy: str = "auto"
    requested_start: float = 0.0
    requested_end: float = 0.0
    actual_start: float = 0.0
    actual_end: float = 0.0
    video: str | None = None          # relative to workdir
    audio: str | None = None          # relative to workdir
    reencode_recommended: bool = False


class AsrInfo(BaseModel):
    status: AsrStatus = "pending"
    backend: str | None = None
    model: str | None = None
    aligner: str | None = None
    chunks_total: int = 0
    chunks_done: int = 0
    chunk_plan: list[tuple[float, float]] = Field(default_factory=list)  # absolute times
    srt: str | None = None
    vtt: str | None = None


class SlideChange(BaseModel):
    time: float                        # segment-relative seconds
    ratio: float = 0.0                 # change ratio at detection


class SlideFrame(BaseModel):
    time: float                        # segment-relative change time
    frame: str                         # image path relative to workdir
    vlm_status: Literal["pending", "done", "vlm_failed"] = "pending"
    description: str | None = None


class SlidesInfo(BaseModel):
    status: SlidesStatus = "pending"
    suspect: bool = False            # change count hit max_frames_per_segment
    changes: list[SlideChange] = Field(default_factory=list)
    frames: list[SlideFrame] = Field(default_factory=list)
    opening_frame: str | None = None  # workdir-relative; opening-slide description source
    srt: str | None = None
    vtt: str | None = None


class Segment(BaseModel):
    id: str
    index: int
    start: float                       # absolute source timeline
    end: float
    duration: float
    status: SegmentStatus = "pending"
    speech_intervals: list[tuple[float, float]] = Field(default_factory=list)  # absolute
    cut: CutInfo | None = None
    asr: AsrInfo = Field(default_factory=AsrInfo)
    slides: SlidesInfo = Field(default_factory=SlidesInfo)

    @property
    def cut_drift(self) -> float:
        """Seconds the cut file starts *before* the requested start
        (keyframe snapping). Cue times are re-based by this amount."""
        if self.cut is None:
            return 0.0
        return self.cut.requested_start - self.cut.actual_start


# ---------------------------------------------------------------------------
# Manifest


class Manifest(BaseModel):
    version: int = 1
    project: str = "video-segment-analysis"
    created: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    source: SourceInfo = Field(default_factory=SourceInfo)
    pipeline: PipelineInfo = Field(default_factory=PipelineInfo)
    vad: VadInfo = Field(default_factory=VadInfo)
    segments: list[Segment] = Field(default_factory=list)

    # -- persistence -------------------------------------------------------

    @classmethod
    def load(cls, workdir: Path) -> "Manifest":
        path = workdir / "manifest.json"
        if not path.is_file():
            raise ManifestError(
                f"no manifest at {path} — run `vsa segment --source ...` first")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.model_validate(data)
        except Exception as exc:
            raise ManifestError(f"manifest corrupt: {path}: {exc}") from exc

    def save(self, workdir: Path) -> None:
        path = workdir / "manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".manifest-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.model_dump(mode="json"), f, indent=2, ensure_ascii=False)
                f.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    # -- helpers -----------------------------------------------------------

    def segment(self, seg_id: str) -> Segment:
        for seg in self.segments:
            if seg.id == seg_id:
                return seg
        raise ManifestError(f"segment not in manifest: {seg_id}")

    def set_stage(self, stage: Stage) -> None:
        """Advance (never regress) the pipeline stage."""
        if stage_rank(stage) > stage_rank(self.pipeline.stage):
            self.pipeline.stage = stage

    def select(self, segment_ids: list[str] | None,
               statuses: set[SegmentStatus] | None = None) -> list[Segment]:
        """Segments filtered by explicit ids and/or required statuses."""
        segs = self.segments
        if segment_ids:
            wanted = set(segment_ids)
            missing = wanted - {s.id for s in segs}
            if missing:
                raise ManifestError(f"unknown segment ids: {sorted(missing)}")
            segs = [s for s in segs if s.id in wanted]
        if statuses:
            segs = [s for s in segs if s.status in statuses]
        return segs

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
