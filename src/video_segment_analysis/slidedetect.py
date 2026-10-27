"""Slide-change detection over sampled frames (DESIGN §9.2).

Pure numpy/cv2: the pipeline streams ``sample_fps`` grayscale frames
(``(N, h, w) uint8``, frame ``i`` ↔ ``i / sample_fps`` segment-relative
seconds) in via ``ffmpegio.sample_frames``; this module only classifies
which samples mark a slide change.

Algorithm (per DESIGN §9.2):
- Per-sample absolute-diff ratio vs the *previous* sample (``prev`` updates
  EVERY sample so sustained animations show up as continuous change runs).
- Suppression window: for ``min_change_interval`` seconds after an accepted
  change all detections are dropped (no capture decisions); the timer
  restarts on each accepted change.
- ``ratio >= change_ratio`` starts/extends a pending candidate; the
  candidate's reference frame tracks the run (the transition's leading
  edge). When change activity stops, the current image is stability-checked
  against that reference; after ``stable_frames`` consecutive stable
  samples the change is ACCEPTED at ``pending.t_start`` (the first frame
  of the run) — provided the settled image is *distinct from the pre-change
  image* (the "stable & distinct from pre-change" test: a single-sample
  flicker recovers to the pre-change image and is discarded).
- A candidate still pending at EOF is accepted (end-of-segment slide).
- The first accepted change is skipped when ``t < 2 s`` (segment-start
  artifact); it still starts the suppression window.
- More than ``max_frames_per_segment`` accepted changes → keep the
  strongest by ratio (sorted by time) and mark the result ``suspect``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import SlidesConfig

log = logging.getLogger("vsa.slides")

#: First accepted change earlier than this is a segment-start artifact.
START_ARTIFACT_T = 2.0
#: frames_close tolerance: fraction of pixels allowed to differ.
CLOSE_TOL = 0.05


@dataclass
class ChangeEvent:
    """An accepted slide change. ``t``: segment-relative seconds (first
    frame of the candidate run); ``ratio``: max changed-pixel ratio of the run."""

    t: float
    ratio: float


@dataclass
class DetectionResult:
    changes: list[ChangeEvent]
    suspect: bool = False


def frames_close(
    a: np.ndarray, b: np.ndarray, tol: float = CLOSE_TOL, *, threshold: int = 28
) -> bool:
    """True when the fraction of pixels differing beyond ``threshold``
    gray levels is below ``tol``."""
    diff = cv2.absdiff(a, b)
    changed_fraction = np.count_nonzero(diff > threshold) / diff.size
    return bool(changed_fraction < tol)


@dataclass
class _Candidate:
    t_start: float
    t_last: float
    frame: np.ndarray      # latest frame of the change run (transition leading edge)
    pre_frame: np.ndarray  # frame immediately before the run started
    max_ratio: float


def detect_changes(frames: np.ndarray, cfg: SlidesConfig) -> DetectionResult:
    """Detect slide changes in ``(N, h, w)`` uint8 frames sampled at
    ``cfg.sample_fps``. Returns accepted events plus a ``suspect`` flag
    set when the ``max_frames_per_segment`` cap fired."""
    n = int(frames.shape[0])
    if n < 2:
        return DetectionResult(changes=[])

    fps = cfg.sample_fps
    threshold = int(cfg.pixel_diff_threshold)
    changes: list[ChangeEvent] = []
    prev = frames[0]
    last_accepted_t = -np.inf
    pending: _Candidate | None = None
    stable_run = 0

    def accept(t: float, ratio: float) -> None:
        nonlocal last_accepted_t
        last_accepted_t = t  # suppression window starts on every acceptance
        if not changes and t < START_ARTIFACT_T:
            log.info("slides: skipping segment-start artifact change at t=%.2fs", t)
            return
        changes.append(ChangeEvent(t=t, ratio=ratio))

    for i in range(1, n):
        t = i / fps
        diff = cv2.absdiff(frames[i], prev)
        ratio = np.count_nonzero(diff > threshold) / diff.size
        pre = prev
        prev = frames[i]

        # ---- min-change interval: suppression window ----
        if t - last_accepted_t < cfg.min_change_interval:
            pending = None  # drop candidate; frames compared, never acted on
            stable_run = 0
            continue

        if ratio >= cfg.change_ratio:
            if pending is None:
                pending = _Candidate(t, t, frames[i], pre, ratio)
            else:  # change activity continuing (fade/wipe/build in progress)
                pending.t_last = t
                pending.frame = frames[i]
                pending.max_ratio = max(pending.max_ratio, ratio)
            stable_run = 0
        elif pending is not None:
            # Change activity stopped: is the new image stable & distinct
            # from the pre-change image?
            if frames_close(frames[i], pending.frame, tol=CLOSE_TOL, threshold=threshold):
                stable_run += 1
                if stable_run >= max(1, cfg.stable_frames):
                    if frames_close(frames[i], pending.pre_frame, tol=CLOSE_TOL, threshold=threshold):
                        log.debug("slides: discarded flicker candidate at t=%.2fs", pending.t_start)
                        pending = None  # recovered to pre-change image (flicker)
                        stable_run = 0
                    else:
                        accept(pending.t_start, pending.max_ratio)
                        pending = None
                        stable_run = 0
            else:
                pending = None  # drifted away → false candidate
                stable_run = 0

    if pending is not None:
        # Candidate still pending at EOF: accepted (end-of-segment slide).
        accept(pending.t_start, pending.max_ratio)

    suspect = False
    if len(changes) > cfg.max_frames_per_segment:
        log.warning(
            "slides: %d changes exceed max_frames_per_segment=%d — keeping the "
            "strongest by ratio and marking the segment slide_detect_suspect "
            "(tune slides.change_ratio / slides.min_change_interval?)",
            len(changes), cfg.max_frames_per_segment,
        )
        changes = sorted(changes, key=lambda c: c.ratio, reverse=True)[
            : cfg.max_frames_per_segment
        ]
        changes.sort(key=lambda c: c.t)
        suspect = True

    log.info("slides: detected %d change(s)%s", len(changes), " (suspect)" if suspect else "")
    return DetectionResult(changes=changes, suspect=suspect)
