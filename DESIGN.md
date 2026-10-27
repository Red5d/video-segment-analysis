# Design Specification — video-segment-analysis (VSA)

**Project name:** `video-segment-analysis` (short: **VSA**)
**Python package:** `video_segment_analysis` · **CLI command:** `vsa`
**Status:** Ready for implementation
**Target language:** Python 3.11+
**Package manager:** [uv](https://docs.astral.sh/uv/) — `pyproject.toml` + committed `uv.lock`; all install/run/test commands go through `uv` (no bare `pip`/`venv`/`python` workflows)
**External runtime dependencies:** `ffmpeg` + `ffprobe` binaries (system install, NOT the Python `ffmpeg` package)

---

## 1. Problem Statement

We have recordings of day-long conference livestreams (multiple hours; often hosted on YouTube, sometimes as local files). The recording contains several presentations separated by a few minutes of silence (breaks, setup, room noise).

We want **three sequential processes**, each its own CLI command operating on a shared output directory:

**Process 1 — Segment & Cut (`vsa segment`):**
1. Obtain audio from a local video file or a YouTube URL (via the `yt-dlp` Python module).
2. Run speech/silence detection over the audio to find intervals containing speech.
3. Group speech intervals into *segments* (one per presentation), separated by long silences.
4. Cut the video into per-segment files (from the local file with `ffmpeg`, or by downloading time ranges from YouTube with `yt-dlp`).

**Process 2 — Transcribe (`vsa transcribe`):**
1. For each cut segment, transcribe its audio with **Qwen3 ASR running locally through the `qwen-asr` Python module**.
2. Write one **SRT/VTT** subtitle file per segment (speech transcript track).

**Process 3 — Slide Detection & Visual Description (`vsa slides`):**
1. For each segment video, sample frames and detect moments where a configurable percentage of the screen changes (default: a large majority, e.g. ≥ 50% of pixels change). For presentation content these correspond to slide changes.
2. A **minimum change interval** (`min_change_interval`, default 3 s) suppresses any further change handling after a change is detected, so on-screen animations don't cause a burst of frame captures.
3. Extract a full-resolution frame for each accepted change.
4. Send each frame to a vision LLM (generic OpenAI-compatible API) and ask it to describe/transcribe the slide's visual content.
5. Write a **second SRT/VTT file per segment** in which each slide's description is a subtitle cue spanning from the slide's appearance until the next slide change (a "slide track").

The result: per presentation segment, a video file, a transcript subtitle track, and a slide-description subtitle track — all on a consistent timeline.

## 2. Goals and Non-Goals

### Goals
- Reliable on 8–12 hour inputs; no requirement to keep the full video in memory.
- Resumable: every stage is idempotent and checkpointed; a crash restarts only the failed unit of work.
- Each process is independently runnable and re-runnable on the same output directory (`slides` can run before `transcribe`, repair runs target one process and one segment).
- ASR runs fully locally/offline via the official `qwen-asr` Python module (transformers backend). **No vLLM, no DashScope, no network calls for ASR.**
- LLM/VLM calls go through a **generic OpenAI-compatible client** so any endpoint works (OpenAI, local llama.cpp/Ollama/vLLM servers, LiteLLM proxies, …), configured by `base_url` + `model` + API-key env var.
- Both source types (local file, YouTube URL) handled behind one interface.
- All tunables (silence thresholds, slide-change percentage, min-change interval, model names, chunk sizes) in one config file with sane defaults.
- Clean output tree with a machine-readable manifest describing every artifact.

### Non-Goals
- Real-time/streaming processing (offline batch only).
- Speaker diarization.
- Support for sources other than local files + sites supported by yt-dlp.
- Editing/rewriting transcript content beyond what ASR returns.
- Web UI. CLI only.
- Running Qwen3-ASR through vLLM or the DashScope API (explicitly out of scope; the `qwen-asr` transformers backend is the ASR path).

## 3. High-Level Architecture

```
        ┌──────────────────────────────────────────────────────────────┐
        │                 shared core package                          │
        │   config · manifest · ffmpeg wrappers · yt-dlp adapter ·     │
        │   timeline algebra · srt/vtt writer · asr/vlm clients · log  │
        └──────────────────────────────────────────────────────────────┘
                    ▲                     ▲                    ▲
   PROCESS 1         │        PROCESS 2    │        PROCESS 3   │
   `vsa segment`  ───┘        `vsa transcribe` ──┘   `vsa slides` ──┘
┌────────────────────────┐ ┌────────────────────────┐ ┌────────────────────────┐
│ 1. resolve source      │ │ 1. read manifest.json  │ │ 1. read manifest.json  │
│    (local | URL)       │ │    (segments with      │ │    (segments with      │
│ 2. extract full audio  │ │     status=cut)        │ │     status=cut)        │
│    → 16k WAV           │ │ 2. cut audio chunks    │ │ 2. sample frames at    │
│ 3. silence/speech      │ │    at silence gaps     │ │    low fps, frame-to-  │
│    detection           │ │ 3. transcribe via      │ │    frame changed-pixel │
│    → speech intervals  │ │    qwen-asr module     │ │    ratio               │
│ 4. group intervals →   │ │    (+ forced aligner   │ │ 3. detect changes w/   │
│    segments            │ │    timestamps)         │ │    3 s min-change      │
│ 5. cut per-segment     │ │ 4. build cues →        │ │    interval            │
│    video + audio       │ │    <seg>.srt / .vtt    │ │ 4. extract full-res    │
│    (manifest.json)     │ │                        │ │    frames → VLM (OpenAI│
└────────────────────────┘ └────────────────────────┘ │    -compatible) →      │
                                                      │    <seg>.slides.srt/.vtt│
                                                      └────────────────────────┘
```

Three independent CLI entry points operating on a shared output directory and a shared `manifest.json` (§10). Each process reads the manifest, does its stage, and writes back its own fields; the per-segment `status` field is the handoff. A convenience command `vsa all` runs all three in order.

### 3.1 Proposed repository layout

```
video-segment-analysis/
├── pyproject.toml              # project "video-segment-analysis"; [project.scripts] vsa = ...
├── uv.lock                     # committed
├── .python-version             # 3.12 (uv pins/creates the env on `uv sync`)
├── README.md
├── config.example.yaml
├── src/video_segment_analysis/
│   ├── __init__.py
│   ├── cli.py                  # typer entry points: segment, transcribe, slides, all
│   ├── config.py               # pydantic-settings models; YAML + env + CLI overrides
│   ├── manifest.py             # Manifest/Segment dataclasses, load/save, state machine
│   ├── timeline.py             # Interval algebra: merge/invert/group/pad/split (pure funcs)
│   ├── sources/
│   │   ├── base.py             # SourceAdapter ABC
│   │   ├── local.py            # LocalFileSource
│   │   └── youtube.py          # YoutubeSource (yt-dlp Python API)
│   ├── ffmpegio.py             # subprocess wrappers: probe, extract_audio, cut, sample_frames, extract_frame
│   ├── vad.py                  # silencedetect parsing → speech intervals (see §6)
│   ├── asr/
│   │   ├── base.py             # AsrClient ABC: transcribe(audio_path) -> list[Utterance]
│   │   ├── qwen_local.py       # DEFAULT: qwen-asr package, transformers backend, forced aligner
│   │   ├── openai_compat.py    # optional: generic OpenAI /audio/transcriptions endpoint
│   │   └── fake.py             # canned outputs for tests / --dry-run
│   ├── vlm/
│   │   ├── base.py             # VlmClient ABC: describe(image_path) -> SlideDescription
│   │   ├── openai_compat.py    # DEFAULT: generic OpenAI-compatible vision chat.completions
│   │   └── fake.py
│   ├── subtitles.py            # Cue dataclass; SRT + WebVTT readers/writers; time formatting
│   ├── segment_pipeline.py     # Process 1 orchestration (vsa segment)
│   ├── transcribe_pipeline.py  # Process 2 orchestration (vsa transcribe)
│   └── slides_pipeline.py      # Process 3 orchestration (vsa slides)
└── tests/
    ├── conftest.py             # fixture factory: synthetic conference video via ffmpeg
    ├── test_timeline.py
    ├── test_vad.py
    ├── test_subtitles.py
    ├── test_cut.py
    ├── test_asr_chunking.py    # chunking + cue builder (stubbed qwen_asr)
    ├── test_slide_detection.py # incl. min_change_interval behavior
    └── test_end_to_end.py      # fake ASR/VLM clients; golden-file SRTs
```

## 4. CLI Surface

```
vsa segment    --source <path-or-url> [--config config.yaml] [--out DIR]
               [--strategy auto|local-copy|local-encode|yt-sections]
               [--segment-id ID ...] [--force] [--dry-run]

vsa transcribe [--config config.yaml] [--out DIR]
               [--backend qwen_local|openai_compat|fake] [--segment-id ID ...]
               [--force] [--dry-run]

vsa slides     [--config config.yaml] [--out DIR]
               [--change-threshold 0.5] [--min-change-interval 3.0]
               [--segment-id ID ...] [--force] [--dry-run]

vsa all        (same options as `segment`, then runs transcribe, then slides)
```

Behavior:
- All commands are invoked through uv so the locked environment is used: `uv run vsa segment …` (documented that way in README; `uv sync` once after clone).
- `--source` is a local path if it exists on disk; a URL if it starts with `http(s)://`. `vsa segment` requires it; `transcribe`/`slides` read it (and everything else) from the manifest.
- Process prerequisites: `transcribe` and `slides` require a manifest with `pipeline.stage >= cut`; `slides` does **not** require transcription to be done (independent tracks).
- Idempotent by default: a stage re-runs only if its outputs are missing/corrupt; `--force` re-runs regardless. `--segment-id` restricts a run to specific segments (used to repair failures).
- `--dry-run` prints the plan (intervals, segments, chunks to transcribe, expected slide-change count, per-segment actions, estimated VLM calls) and exits 0.
- Exit codes: `0` success; `1` fatal (config, source, audio extraction); `2` completed with per-segment failures (details in manifest + summary table printed at end).
- Every long-running step logs progress (e.g. `silence-detect: 3:24:10 / 8:12:00`, `transcribe: seg_003 chunk 12/17`).

## 5. Configuration

One YAML file (`config.yaml`, example committed as `config.example.yaml`), every key overridable by env var `VSA__<SECTION>__<KEY>` and by CLI flags. Secrets (API keys) come from env vars only, never the YAML.

```yaml
source:
  type: auto                    # auto | local | youtube
  cookiefile: null              # yt-dlp cookies.txt for age/privately-gated streams
  video_format: "bv*[height<=1080]+ba/b"   # yt-dlp format string for full-media download

paths:
  workdir: ./out                # everything lives under here (see §10)

audio:
  sample_rate: 16000
  channels: 1
  format: wav                   # pcm_s16le

vad:
  engine: ffmpeg                # ffmpeg (silencedetect) — v1; silero-vad reserved as engine 'silero'
  noise_db: -35                 # dB level considered "silence" (see §6.2 tuning notes)
  min_silence_duration: 2.0     # s: a silence shorter than this is treated as speech (not emitted)

segmentation:
  min_silence_gap: 45.0         # s: silence ≥ this ends a segment (break between talks)
  min_segment_duration: 120.0   # s: speech runs shorter than this are dropped as noise
  pad_start: 3.0                # s: extend segment start backwards into preceding silence
  pad_end: 5.0                  # s: extend segment end forward (Q&A tail, applause)
  merge_speech_gap: 2.0         # s: speech intervals closer than this are merged

cut:
  strategy: auto                # auto | local-copy | local-encode | yt-sections (see §7.3)
  video_format_local: "copy"    # stream copy for local cutting

asr:
  backend: qwen_local           # qwen_local (default) | openai_compat | fake
  # --- qwen_local: official qwen-asr Python package, transformers backend ---
  model: Qwen/Qwen3-ASR-1.7B    # HF id or local path; Qwen/Qwen3-ASR-0.6B for low VRAM
  device: cuda:0                # cpu works but is slow (documented)
  dtype: bfloat16               # bfloat16 | float16 | float32
  aligner: Qwen/Qwen3-ForcedAligner-0.6B  # null → chunk-level timing only (§8.4)
  aligner_device: null          # defaults to `device`
  language: null                # null = auto-detect; e.g. "English" to force
  max_new_tokens: 1024          # raise for long chunks
  max_inference_batch_size: 8   # chunk batch size inside qwen_asr (lower = less VRAM)
  attn_implementation: sdpa     # flash_attention_2 optional
  # --- chunking (applies to all backends) ---
  chunk_target_seconds: 120.0   # prefer utterance chunks near this size
  max_chunk_seconds: 280.0      # hard cap: forced aligner supports ≤ 5 min per alignment
  # --- openai_compat backend only (generic OpenAI-compatible endpoint) ---
  openai_base_url: http://localhost:8000/v1
  openai_model: whisper-1       # whatever the endpoint names its ASR model
  openai_api_key_env: VSA_ASR_API_KEY   # omitted from request if unset
  # --- shared request behavior ---
  concurrency: 1                # >1 only meaningful for openai_compat (local model is 1 process)
  retries: 2
  retry_backoff: [2, 8, 30]     # seconds between attempts
  request_timeout: 600

subtitles:
  formats: [srt, vtt]
  max_cue_duration: 10.0        # s: cues are split at sentence boundaries past this
  max_cue_chars: 42             # soft wrap for readability
  max_cue_gap: 1.5              # s: word timestamp gap that forces a cue break

slides:
  sample_fps: 1.0               # frames analyzed per second per segment
  analyze_width: 320            # px width frames are downscaled to for diffing
  pixel_diff_threshold: 28      # 0–255 gray-level delta for a pixel to count as "changed"
  change_ratio: 0.5             # fraction of pixels that must change to flag a slide change
  min_change_interval: 3.0  # s: after an accepted change, ignore all further change
                                #    detections (no frame capture/analysis decisions) for this
                                #    long — absorbs on-screen animations/transitions (§9.2)
  stable_frames: 2              # new visual must persist N consecutive samples before acceptance
  frame_offset: 1.5             # s: extract frame at change_time + offset (transitions finish)
  max_frames_per_segment: 200   # safety cap against noise (warn when hit)
  crop: null                    # optional "x,y,w,h" to mask out overlays (channel logo, lower thirds)

vlm:
  backend: openai_compat        # openai_compat (default) | fake
  model: gpt-4o                 # any vision model the endpoint serves, e.g.
                                # qwen2.5-vl-7b-instruct (vLLM/llama.cpp/Ollama /v1)
  base_url: https://api.openai.com/v1   # point at any OpenAI-compatible server
  api_key_env: VSA_VLM_API_KEY  # key read from this env var; omitted if unset (local servers)
  image_format: jpeg
  image_quality: 85
  prompt_file: null             # custom prompt; built-in default otherwise (§9.4)
  concurrency: 2                # ThreadPoolExecutor size
  retries: 3
  retry_backoff: [2, 8, 30]
  request_timeout: 120

logging:
  level: INFO
  file: null                    # optional log file in workdir
```

## 6. Speech / Silence Detection (Process 1)

### 6.1 Audio acquisition
- **Local source:** `ffmpeg -hide_banner -nostats -i <video> -vn -ac 1 -ar 16000 -c:a pcm_s16le audio.wav`
- **YouTube source:** download audio once with the yt-dlp Python API:
  ```python
  ydl_opts = {
      "format": "bestaudio/best",
      "postprocessors": [{"key": "FFmpegExtractAudio",
                          "preferredcodec": "wav",
                          "preferredquality": "0"}],
      "outtmpl": str(workdir / "audio.%(ext)s"),
      "noplaylist": True,
      "quiet": True,
  }
  ```
  Rationale: for an hours-long stream, one full audio download (~100–300 MB WAV) is far cheaper and more robust than per-segment manifest fetches. Per-segment *video* downloads remain supported (§7.3) for building the final clip files; audio for VAD/ASR always comes from the single full-audio pass.
- Validate with `ffprobe` that the WAV duration matches the source duration within ±2 s (record the value in the manifest as `drift_warning`).

### 6.2 Silence detection engine
Primary: ffmpeg's `silencedetect` filter on the 16 kHz mono WAV:
```
ffmpeg -hide_banner -nostats -i audio.wav \
  -af silencedetect=noise=-35dB:d=2 -f null -
```
Parse stderr lines:
```
[silencedetect @ ...] silence_start: 3721.44
[silencedetect @ ...] silence_end: 3899.12 | silence_duration: 177.68
```
The parser must:
- Handle a trailing open-ended `silence_start` (silence to EOF) by closing it at file duration.
- Drop silences with `duration < min_silence_duration` (filter artifact noise).
- Raise a clear error if zero silences are found (likely threshold misconfiguration → suggest tuning `noise_db` / `min_silence_duration`, or `--dry-run` histogram).

`vad.py` exposes `detect_silences(wav_path, cfg) -> list[Interval]` and unit tests parse captured ffmpeg stderr fixtures.

Tuning notes (document in README): conference audio with applause/floor noise may need `noise_db: -30`; very quiet rooms `-40`. Applause is non-speech energy but silencedetect counts it as noise; that's acceptable because §6.3 grouping absorbs it into segment boundaries/pads.

(Reserved: a `silero` engine using silero-vad behind the same ABC. Not in v1 scope.)

### 6.3 Interval algebra (`timeline.py`)
Pure, fully unit-tested functions over `(start: float, end: float)` tuples on the **absolute source timeline**:
- `invert(intervals, total_duration)` → speech intervals from silences.
- `merge(intervals, max_gap)` → merge intervals separated by gap < `merge_speech_gap`.
- `drop_short(intervals, min_dur)` → discard speech runs < `min_segment_duration` (jingles, "can you hear me", room chatter).
- `group_runs(intervals, gap)` → split into segment-level runs where gaps ≥ `min_silence_gap` (these gaps are the inter-talk breaks).
- `pad(intervals, before, after, total_duration)` → extend boundaries, clamped to `[0, total_duration]`, merging intervals that overlap after padding.
- `split_at(intervals, points)` and `split_even_cap(interval, cap, candidate_points)` → for ASR chunking (§8.3).

### 6.4 Segmentation output
Segments are numbered from `001`. The stage writes the manifest (§10) with `status=segmented`. No media is produced by this stage.

## 7. Process 1 — Segment & Cut (`vsa segment`)

### 7.1 Stage graph
```
resolve_source → extract_audio → detect_speech → segment → cut_media → finalize
```
Each stage persists its outputs and manifest state; `vsa segment` resumes from the first incomplete stage. It produces: `manifest.json`, `audio.wav`, per-segment `<seg>.mp4` + `<seg>.wav`, and per-segment speech-interval lists (persisted in the manifest for the transcribe process's chunking). It does **not** call any LLM/ASR.

### 7.2 Source adapters (`sources/base.py`)
```python
class SourceAdapter(ABC):
    def probe(self) -> MediaInfo: ...                    # duration, streams, has_video
    def extract_full_audio(self, dst_wav: Path) -> None: ...
    def cut_segment(self, seg: Segment, dst_video: Path, dst_audio_wav: Path,
                    strategy: str) -> CutResult: ...     # returns actual file timespans
```
`LocalFileSource` and `YoutubeSource` implement it; `resolve_source()` picks by `--source` string and `source.type` config.

### 7.3 Cutting strategies (`cut_segment`)

| strategy | media source | how | precision | cost |
|---|---|---|---|---|
| `local-copy` | local file | `ffmpeg -ss <start> -to <end> -i in.mp4 -c copy -avoid_negative_ts make_zero seg.mp4` (`-ss` **before** `-i` for fast seek) | start snaps to previous keyframe (sub-second to ~2 s early); audio stream copied | near-zero |
| `local-encode` | local file | same seek + `-c:v libx264 -preset veryfast -crf 22 -c:a aac` | frame-accurate | full transcode |
| `yt-sections` | YouTube | yt-dlp Python API, per segment: `{"download_ranges": [f"*{mm:ss}-{mm:ss}"], "force_keyframes_at_cuts": True, "outtmpl": ".../seg_%03d.%(ext)s"}` | accurate (keyframes forced via re-encode) | refetch + re-encode per segment |
| `auto` (default) | — | local file → `local-copy`; YouTube → `full-download` (see below) | | |

**Auto behavior:**
- **Local:** `local-copy`. After cutting, `ffprobe` the result: if its duration deviates from requested by > 3 s (sparse keyframes), log a warning and offer `--force` re-run with `local-encode` for that segment. The measured actual span (`CutResult.actual_start/actual_end`, derived from ffprobe + the seek offset) is stored in the manifest and used to re-base subtitle times (§8.5) so cues stay aligned even when the cut starts on an earlier keyframe.
- **YouTube:** default is **full download once** (`yt-dlp` full video at `source.video_format`, then `local-copy` cuts). This is the robust choice for hours-long files: one sequential download, no per-segment manifest/segment re-fetching (which risks throttling/bans and per-cut re-encode of keyframe boundaries).
- `yt-sections` is opt-in via `--strategy yt-sections` for users who explicitly don't want the full video on disk. Time-range syntax requires the `*` prefix (e.g. `*1:02:03-1:15:00`); note it needs ffmpeg and re-encodes when `force_keyframes_at_cuts` is set. Document that live streams need the stream to have ended before section downloads work.
- Per-segment audio WAV for ASR is always cut from the full 16 kHz WAV with `ffmpeg -ss/-to -i audio.wav ... seg_001.wav` (sample-accurate, cheap, identical for all strategies).

### 7.4 Segment-process manifest state
On success, pipeline stage advances `initialized → audio_ready → speech_detected → segmented → cut`, and each segment gets `status: cut` with its cut fields (files, actual timespan, speech intervals) populated. Failure states `source_failed`, `audio_failed`, `cut_failed` are re-enterable by the same command.

## 8. Process 2 — Transcribe (`vsa transcribe`)

### 8.1 Stage graph and inputs
```
load_manifest → plan_chunks → transcribe_chunks → build_cues → write_subtitles → finalize
```
Inputs: manifest segments with `status: cut` (skips `transcribed` segments; `--segment-id` targets repairs; `--force` redoes). For each segment it cuts ASR audio chunks from `<seg>.wav` at the persisted speech-interval gaps, runs ASR, builds cues, writes `<seg>.srt`/`<seg>.vtt`, and advances per-segment `status: cut → transcribing → transcribed` (failure `transcribe_failed` re-enterable). Transcription of hours of audio is the pipeline's long pole: chunks across all segments are queued first-come-first-served so the GPU never idles between segments.

### 8.2 ASR clients

**Client ABC (`asr/base.py`):**
```python
@dataclass
class Word:
    text: str
    start: float
    end: float

@dataclass
class Utterance:
    start: float      # seconds, RELATIVE TO THE SUBMITTED AUDIO
    end: float
    text: str
    words: list[Word] | None = None   # None when no aligner / API lacks timestamps

class AsrClient(ABC):
    def transcribe(self, audio_path: Path, language: str | None) -> list[Utterance]: ...
```

**Default backend — `qwen_local` (`asr/qwen_local.py`).** Uses the official **`qwen-asr`** PyPI package (import `qwen_asr`) with its **transformers backend**. No server, no vLLM, no DashScope; weights auto-download from Hugging Face on first use (or use a local path). One model instance is loaded once per process and reused across all chunks/segments:

```python
import torch
from qwen_asr import Qwen3ASRModel

model = Qwen3ASRModel.from_pretrained(
    cfg.asr.model,                          # "Qwen/Qwen3-ASR-1.7B"
    dtype=torch.bfloat16,
    device_map=cfg.asr.device,              # "cuda:0"
    max_inference_batch_size=cfg.asr.max_inference_batch_size,
    max_new_tokens=cfg.asr.max_new_tokens,
    forced_aligner=cfg.asr.aligner,         # "Qwen/Qwen3-ForcedAligner-0.6B" (optional)
    forced_aligner_kwargs=dict(dtype=torch.bfloat16, device_map=cfg.asr.aligner_device),
)

results = model.transcribe(audio=[chunk_path, ...],   # local WAV paths, batched
                           language=None,             # auto-detect; "English" to force
                           return_time_stamps=True)   # requires aligner
# each result: .text, .language, .time_stamps (word/char level, seconds)
```

Mapping to the ABC: one chunk → one `Utterance` with `words` populated from `time_stamps` (chunk-relative seconds). Model/aligner load failures produce a startup error with clear guidance (VRAM, `device`, model download). The `0.6B` model is the documented low-VRAM option; `device: cpu` is allowed (slow; warn).

**Optional backend — `openai_compat` (`asr/openai_compat.py`).** Generic OpenAI-compatible transcription: `POST {base_url}/audio/transcriptions` (multipart, WAV file + `model`), via the `openai` SDK's `audio.transcriptions.create`. Returns text only → cues get chunk-window timing (§8.4, no-word path). Works against any server implementing that route (local llama.cpp/whisper.cpp servers, faster-whisper-xy, LiteLLM, hosted providers). `api_key_env` key omitted from the request when the env var is unset (typical for local servers).

**`fake` backend.** Reads canned `Utterance` lists from a JSON fixture (tests, `--dry-run` plumbing).

### 8.3 Chunking (all backends)
The forced aligner accepts **≤ 5 minutes of speech per alignment**, and long single generations degrade quality, so:

1. For a segment `[a, b]`, take its persisted speech intervals and cut them into **utterance chunks**: consecutive speech intervals are merged greedily into one chunk while `chunk_duration ≤ max_chunk_seconds` (280 s) and near `chunk_target_seconds` (120 s); a chunk boundary is always placed inside an inter-interval silence gap, so chunks never truncate speech. Chunks tile the padded segment span (silence gaps are included inside chunks; the first chunk starts at the first speech interval, the last ends at the last one).
2. Each chunk's audio is cut from the segment WAV. For `qwen_local`, chunks are transcribed in batches of `max_inference_batch_size` (chunks of different segments may share a batch). For `openai_compat`, requests run with `concurrency` and `retries`/`retry_backoff` on transient errors.
3. Chunk completion is checkpointed in the manifest (`asr.chunks_done`); a chunk failure after retries marks the segment `transcribe_failed` (partial SRTs kept as `<seg>.srt.partial`, never counted complete).

### 8.4 Timing model (the critical design point)
- **With aligner (`qwen_local` + `aligner`):** transcription yields word/character timestamps. The cue builder (§8.5) groups words into cues: break at sentence-final punctuation, when `cue.duration > max_cue_duration`, when `next.word.start − prev.word.end > max_cue_gap`, or at `max_cue_chars`. Cue start/end are the first/last word timestamps → accurate, speech-anchored SRT.
- **Without aligner (or `openai_compat`):** the chunk's text is one coarse `Utterance(chunk_start, chunk_end)`; the cue builder splits at sentence boundaries and allocates time **proportionally by character count** within the chunk window (monotonic, clamped so cues never overlap). Documented as ±seconds-accurate.
- Aligner language note: `Qwen3-ForcedAligner-0.6B` supports 11 languages (en, zh, yue, fr, de, it, ja, ko, pt, ru, es). For other detected languages, emit the no-word chunk-window path for that chunk and log once per segment.

### 8.5 Cue building and subtitle writing

- Cue times start in **segment-relative** coordinates (ASR chunk times are already relative to the segment start; subtract `actual_start − requested_start` drift recorded by the cut in §7.3 so cues align with the cut file).
- `subtitles.py` implements `write_srt(path, cues)` / `write_vtt(path, cues)` from a `Cue(index, start, end, text)` list:
  - SRT: `HH:MM:SS,mmm --> HH:MM:SS,mmm`, CRLF, UTF-8 (no BOM), text wrapped at `max_cue_chars`.
  - VTT: `WEBVTT` header, `HH:MM:SS.mmm` dots.
  - Enforce non-overlap (`prev.end ≤ next.start`, min gap 0.02 s); clamp to `[0, segment_duration]`.
  - Round-trip tests: write → ffmpeg `-f srt`/`-f webvtt` parse succeeds; golden-file comparisons.
- Outputs: `<seg>.srt` and `<seg>.vtt` (both, per `subtitles.formats`).

## 9. Process 3 — Slide-Change Detection and Visual Description (`vsa slides`)

### 9.1 Frame sampling (`ffmpegio.sample_frames`)
Decode at `slides.sample_fps` directly in ffmpeg, piped to numpy (no per-frame temp files):
```
ffmpeg -hide_banner -loglevel error -i seg_001.mp4 \
  -vf "fps=1[,crop=x:y:w:h],scale=320:-2,format=gray" -pix_fmt gray -f rawvideo -
```
Each raw frame is `analyze_width × analyze_height` bytes. Store as a memory-mapped `(N, h, w)` uint8 array. Frame i corresponds to timestamp `i / sample_fps` (segment-relative).

Cost check: a ~1 h segment at 1 fps ≈ 3.6k frames × ~55 KB ≈ 200 MB gray; process segments sequentially and stream in row blocks if N is large.

### 9.2 Change detection algorithm with min-change interval (`slides/detector.py`)
Pure function, unit-tested on synthetic frame sequences:

```python
def detect_changes(frames, cfg) -> list[float]:
    prev = frames[0]
    last_accepted_t = -inf          # time of the last ACCEPTED change
    pending = None                  # candidate change awaiting stability proof
    for i in 1..N-1:
        t = i / sample_fps
        changed = (frames[i] - prev).abs() > cfg.pixel_diff_threshold
        ratio = changed.mean()      # fraction of pixels that changed
        prev = frames[i]

        # ---- MIN-CHANGE INTERVAL: suppression window ----
        if t - last_accepted_t < cfg.min_change_interval:   # default 3.0 s
            pending = None          # drop any candidate; no capture decisions taken
            continue                # frames are still compared (cheap), never acted on

        if ratio >= cfg.change_ratio:
            if pending is None:
                pending = Candidate(t_start=t, t_last=t, frame=i)
            else:
                pending.t_last = t  # change activity continuing (fade/wipe in progress)
        elif pending is not None:
            # change activity stopped: is the new image stable & distinct from pre-change?
            if frames_close(frames[i], pending.frame, tol=0.05) :
                stable_run += 1
                if stable_run >= cfg.stable_frames:
                    emit(pending.t_start)          # ACCEPTED; starts a new suppression window
                    last_accepted_t = pending.t_start
                    pending = None
            else:
                pending = None      # drifted away → false candidate
    # a candidate still pending at EOF is accepted (end-of-segment slide)
```

Required behaviors, restated as acceptance criteria:
- **Min-change interval (`slides.min_change_interval`, default 3.0 s):** immediately after a change is accepted, no further change may be accepted, and no frame is scheduled for capture, until `min_change_interval` seconds have elapsed. Any change detections inside the window are discarded (they are animation/transition frames). This prevents extracting a burst of frames from a single animated transition and bounds worst-case frame extraction to `1 / min_change_interval` per second. The timer restarts on each accepted change.
- `prev` updates every sample, so gradual animations (fades, wipes, bullet builds) show up as sustained change activity; the `stable_frames` requirement accepts a slide when the image becomes stable **after** the transition, and the emitted timestamp is the first frame of the candidate run.
- If change activity is still ongoing when the suppression window expires (e.g. a > 3 s animation), a new candidate starts at the moment the window expires; the emitted time is that expiry, so the extracted frame is taken on the tail of the transition (acceptable; `frame_offset` helps).
- The first accepted change is skipped if it's at `t < 2 s` (segment-start artifact).
- Cap: if more than `max_frames_per_segment` changes are detected, keep the strongest by change ratio, log a loud warning, and mark the segment `slide_detect_suspect` in the manifest (don't silently flood the VLM).
- Config semantics to document in README: `change_ratio: 0.5` (default) requires a majority of the screen to change — good for whole-slide replacements (title slide → content slide with images). Text-only slides with similar layouts often change only 15–35% of pixels; lower `change_ratio` to ~0.2 (with `pixel_diff_threshold` ~30) to catch bullet-level builds. `min_change_interval` should be raised (e.g. 5–8 s) for heavily animated decks.

### 9.3 Frame extraction
For each accepted change time `t`, extract one full-resolution frame at `min(t + frame_offset, segment_duration − 0.5)`:
```
ffmpeg -hide_banner -loglevel error -ss <t+offset> -i seg_001.mp4 \
  -frames:v 1 -q:v 2 seg_001_slide_014.jpg
```
(Seek-before-input fast seek + JPEG; re-extract with accurate seek `-i … -ss` after input if the fast-seek frame is more than `frame_offset + 1 s` off target.) Frames are saved under `<seg>/slides/` and referenced in the manifest. Apply the same optional `crop` as §9.1 to mask logos/lower-thirds.

### 9.4 VLM description (`vlm/openai_compat.py`)
- **Generic OpenAI-compatible client** (the `openai` Python SDK with configurable `base_url`): `chat.completions.create` with one `image_url` content part (base64 JPEG data URL) + text prompt. API key read from `vlm.api_key_env`; omitted when unset (local servers like llama.cpp/Ollama/vLLM). Works with any endpoint that implements OpenAI vision chat (OpenAI, OpenRouter, LiteLLM, vLLM/llama.cpp/Ollama `/v1`).
- Default prompt (overridable via `prompt_file`):

  > You are given one frame from a conference presentation, captured at the moment the slide changed. Describe the slide content for someone who cannot see it: transcribe all visible text verbatim (title first, then bullets in order), describe charts/diagrams and what they show, and mention speaker name/affiliation if visible on the slide. Output plain text, no markdown, no commentary about the image quality. Keep under 250 words.

- Response is cleaned (collapse whitespace, strip surrounding quotes) and stored as the cue text. Retries/backoff on 429/5xx/timeout; a failed frame yields a placeholder cue `"[slide content unavailable]"` and marks the frame `vlm_failed` for `--segment-id` repair runs.
- Concurrency per `vlm.concurrency` (be conservative — vision calls are the slow/expensive part).
- `fake` backend returns fixture text per frame (tests).

### 9.5 Slide subtitle track
Build cues from sorted accepted change times `t_1 < t_2 < … < t_n` (segment-relative):
- cue i: `start = t_i + frame_offset` (clamped ≥ 0), `end = t_{i+1}` (last cue ends at segment end).
- Enforce `end ≥ start + min_cue_duration (1.0 s)`; if violated (rapid consecutive changes surviving the min-change interval), drop the short cue's frame and keep the later one.
- The first slide (before any detected change) gets a cue `[0, t_1)` only if `t_1 > 5 s`, with a VLM description of the opening frame at t≈1 s (title slide) — this captures the talk's opening slide.
- Long descriptions are split into consecutive cues with time allocated proportionally within the slide's window; cues never exceed the window.
- Outputs: `<seg>.slides.srt` and `<seg>.slides.vtt`.
- Manifest: per-segment `slides.status: pending → detecting → extracting → describing → done` (failure states `detect_failed`, `describe_failed` re-enterable).

## 10. Output Layout & Manifest

```
out/
├── manifest.json
├── audio.wav                        # full-source audio (kept; deletable with --cleanup)
├── source_video.mp4                 # only when strategy downloads full media
├── segments/
│   ├── seg_001.mp4                  # cut video
│   ├── seg_001.wav                  # 16k mono audio for ASR
│   ├── seg_001.srt                  # transcript track
│   ├── seg_001.vtt
│   ├── seg_001.slides.srt           # slide-description track
│   ├── seg_001.slides.vtt
│   └── seg_001/slides/slide_014.jpg # extracted frames
└── logs/vsa.log
```

`manifest.json` (canonical source of truth; written atomically via temp-file + rename; every process reads the whole manifest and updates only its own fields):

```json
{
  "version": 1,
  "project": "video-segment-analysis",
  "created": "2026-10-27T09:00:00Z",
  "source": {"type": "youtube", "url": "https://youtu.be/…", "title": "…",
              "duration": 29520.4, "downloaded_full": true},
  "pipeline": {"stage": "cut", "config_hash": "sha256:…"},
  "vad": {"engine": "ffmpeg", "noise_db": -35,
           "speech_intervals": [[312.0, 4210.5], [4390.1, 7001.9], "…"]},
  "segments": [
    {
      "id": "seg_001", "index": 1,
      "start": 309.0, "end": 4213.5, "duration": 3904.5,
      "status": "cut",
      "speech_intervals": [[312.0, 4210.5]],
      "cut": {"strategy": "local-copy", "actual_start": 307.84, "actual_end": 4213.9,
               "video": "segments/seg_001.mp4", "audio": "segments/seg_001.wav"},
      "asr": {"status": "pending", "backend": "qwen_local",
               "model": "Qwen/Qwen3-ASR-1.7B", "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
               "chunks_total": 17, "chunks_done": 0, "srt": null, "vtt": null},
      "slides": {"status": "pending", "changes": [], "frames": [], "srt": null, "vtt": null}
    }
  ]
}
```
Rules:
- All times in the manifest are **absolute source timeline** except fields documented as segment-relative (cue times are never persisted; SRT/VTT files are the record).
- Segment `status` progression: `pending → cut` (segment process) `→ transcribing → transcribed` (transcribe process); the slides process tracks its own `slides.status` and only requires `status ≥ cut`. A segment is fully done when `status == transcribed` and `slides.status == done`.
- `pipeline.stage` is the minimum across the above enums (`initialized < audio_ready < speech_detected < segmented < cut < transcribed < done`), so `vsa transcribe`/`vsa slides` can check prerequisites with one comparison.
- `config_hash` mismatch on a re-run triggers a warning: results may be stale for changed tuning parameters (`--force` to redo).
- `--dry-run` renders per-segment progress as a table across all three processes.

## 11. Cross-Cutting Requirements

- **Resumability:** every stage checks for existing valid outputs (file exists, non-zero, ffprobe-parseable / SRT-parseable) and skips them. No process may rewrite an artifact produced by another process. No stage rewrites outputs it didn't produce unless `--force`.
- **Retries:** all network calls (yt-dlp download, VLM, optional `openai_compat` ASR) wrapped in a shared `retry_with_backoff(fn, retries, backoff, retry_on=(RateLimitError, Timeout, TransientHTTP))` helper.
- **Concurrency:** `concurrent.futures.ThreadPoolExecutor` for VLM and any API-based ASR. The local `qwen_local` ASR is a single GPU-resident model: one instance, batched `transcribe()` calls, **no threads around the model object** (serialize access with a lock).
- **Validation:** after every ffmpeg/yt-dlp command: nonzero exit → capture last 40 lines of stderr into the log and raise `MediaError` with the command (secrets redacted).
- **Logging:** stdlib `logging`, structured prefix per stage (`[vad]`, `[cut:seg_003]`, `[asr:seg_003/chunk_07]`, `[slides:seg_003]`), optional rotating file in `out/logs/`.
- **No secrets in files:** API keys read from env at client construction; a key required by the config (e.g. remote VLM base_url under api.openai.com/api.openrouter) with the env var unset is a startup validation error naming the exact env var.
- **Resource safety:** stream ffmpeg stdout for frame sampling (never decode full video at full res); delete partial files on failure; `--cleanup` flag removes `audio.wav`/`source_video.mp4` after all processes report `done`.

## 12. Dependencies & Environment (uv)

The project is a uv-managed application: `uv init` scaffold (package layout `src/video_segment_analysis`), deps added with `uv add`, dev deps with `uv add --dev`. `uv.lock` and `.python-version` are committed; CI and contributors use `uv sync --frozen`. Never invoke `pip`/`venv` directly.

`pyproject.toml` skeleton:

```toml
[project]
name = "video-segment-analysis"
requires-python = ">=3.11"
dependencies = [
    # core
    "yt-dlp>=2025.1",
    "numpy>=1.26",
    "opencv-python-headless>=4.9",   # frame diffs (cv2.absdiff); alternative: pure numpy
    "openai>=1.40",                  # generic OpenAI-compatible client (VLM + optional ASR)
    "pydantic>=2.7", "pydantic-settings>=2.2",
    "pyyaml>=6.0",
    "typer>=0.12",
    "tenacity>=8.2",
    "rich>=13.0",                    # progress bars & summary tables
    # ASR (default backend) — official Qwen3-ASR toolkit, transformers backend ONLY.
    # Do NOT add the [vllm] extra (out of scope per requirements).
    "qwen-asr>=0.1",                 # import name `qwen_asr`; pulls torch/transformers
]

[project.scripts]
vsa = "video_segment_analysis.cli:app"

[dependency-groups]
dev = ["pytest>=8", "pytest-cov", "responses"]   # HTTP mocks
```

Workflow commands (documented in README):
```bash
uv sync                    # create/refresh .venv from uv.lock (downloads Python 3.12 per .python-version)
uv run vsa segment --source …
uv run vsa transcribe
uv run vsa slides
uv run pytest              # dev loop: uv run pytest tests/test_slide_detection.py -k min_change_interval
uv lock --upgrade-package yt-dlp   # yt-dlp ages quickly; upgrade periodically
```
Optional perf extra (documented, not locked by default): `flash-attn` — install manually per the qwen-asr README (needs `--no-build-isolation`; bf16 only).

System: `ffmpeg`, `ffprobe` on `PATH` (checked at startup, version ≥ 5.0 recommended).
Hardware note (README): `qwen_local` runs on CPU (slow) or CUDA; `Qwen3-ASR-0.6B` for limited VRAM; first run downloads weights (~GB) to the HF cache.

## 13. Testing Strategy

All tests run via `uv run pytest` (dev group deps from §12; CI: `uv sync --frozen && uv run pytest`).

1. **Unit:** `timeline.py` (merge/invert/group/pad/split edge cases: touching intervals, clamp at 0/duration, empty input); `vad.py` (parse captured ffmpeg stderr fixtures incl. open-ended silence, noise lines); `subtitles.py` (round-trip, overlap clamping, wrapping, unicode); change detector on synthetic `ndarray` sequences — required cases: instant cut accepted at correct time; fade (5 samples of change + stable) accepted after stabilization; **animations shorter than `min_change_interval` produce exactly one change**; repeated animations spaced closer than `min_change_interval` produce one change per suppression window; flicker noise (single-sample spikes) rejected by `stable_frames`; cursor blink / small overlay change below `change_ratio` ignored; candidate pending at EOF accepted.
2. **ASR adapter tests:** `qwen_local` is a thin wrapper around `qwen_asr.Qwen3ASRModel` — monkeypatch the class with a stub returning canned `.text`/`.time_stamps`; assert chunking (≤ `max_chunk_seconds`, boundaries in silence gaps), word→cue grouping rules, and the no-aligner proportional path. `openai_compat` tested against a mocked OpenAI-compatible HTTP server (multipart shape, auth header omitted when no key, retries on 429).
3. **Fixture generator (`conftest.py`):** synthesize a miniature conference video with ffmpeg — 6 slides (color+text via `drawtext`) per 2 "talks", talk audio = sine "voice" segments separated by silence gaps of 60 s (between talks) and 3 s (within talk), rendered with `-f lavfi`/filter_complex. Deterministic, ~2 min total, generated in `tests/fixtures/`.
4. **Integration (per process, on the fixture):**
   - `vsa segment` with `vad.engine=ffmpeg` → assert 2 segments at expected times, cut media probeable, manifest `status=cut`, speech intervals persisted.
   - `vsa transcribe` with `asr.backend=fake` → golden SRTs match; re-running is a no-op; `--segment-id` repair works.
   - `vsa slides` with `vlm.backend=fake` → assert 6 slide changes per talk within ±2 s, slide SRT cue windows tile the segment. Also assert slides runs successfully **before** transcribe.
5. **VLM client contract tests:** against a mocked OpenAI-compatible server (image data-URL encoding, prompt content, retry/backoff, key redaction in logs).
6. **YouTube adapter tests:** pure-logic only (opts construction, section string formatting `*H:MM:SS-H:MM:SS`, resume detection); network calls marked `@pytest.mark.network` and excluded from CI default.
7. **Acceptance (manual, documented):** run on a real 2 h+ conference recording; verify segments align to talks, SRT drift < 2 s at talk end, slide cues switch at slide changes in a player.

## 14. Milestones

| # | Deliverable | Definition of done |
|---|---|---|
| M1 | Core skeleton: uv project init, config, manifest, timeline, ffmpegio, sources (local), CLI `segment` | Fixture: audio extract → speech intervals → segments in manifest |
| M2 | Cutting (all local strategies) + subtitles writer module + CLI `transcribe`/`slides` with `fake` clients | Fixture: golden SRTs via `vsa transcribe`; slide SRTs via `vsa slides`; process handoff via manifest `status` |
| M3 | `vsa transcribe` real ASR: `qwen_local` (qwen-asr module + forced aligner) + chunking + cue builder | Live run on short real video produces word-timed SRTs |
| M4 | `openai_compat` ASR + `youtube` source (full-audio/full-download) + `yt-sections` + resume/`--segment-id` repair | Interrupted runs of every process resume cleanly; slides-before-transcribe verified |
| M5 | `vsa slides`: sampler + detector **with min-change interval** + frame extraction + slide SRTs | Fixture slide-change + min-change-interval assertions pass |
| M6 | VLM (OpenAI-compatible) integration + slide-track timing polish + `vsa all` | End-to-end demo on real conference recording |
| M7 | Docs: README quickstart, tuning guide (VAD dB, `change_ratio`, `min_change_interval`), examples | — |

## 15. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| `silencedetect` misfires on applause/room noise (real conferences are noisy) | Tunable `noise_db`/`min_silence_duration`; grouping absorbs applause into segment ends; README tuning guide; future `silero` engine |
| Local Qwen3-ASR needs GPU / VRAM | `Qwen3-ASR-0.6B` + float16 + `max_inference_batch_size: 1–2`; CPU supported (warn on startup); chunk batching caps peak memory |
| Forced aligner: 5 min / 11-language limits | Chunk cap 280 s; non-supported languages fall back to chunk-window timing; both paths covered by cue builder |
| No timestamps when aligner is off / generic ASR API returns text only | Silence-aligned chunk cues + proportional sentence splitting (documented accuracy) |
| Stream-copy cuts start on keyframes → timeline drift | Measured `actual_start` drift stored per segment and subtracted from cue times (§7.3, §8.5) |
| Slide changes on static text slides change < 50% of pixels | Documented knobs (`change_ratio`, `pixel_diff_threshold`); default honors the stated requirement (majority change); detector reports detection count so users can tune |
| Animated transitions produce frame bursts / wasted VLM calls | **`min_change_interval: 3.0 s`** suppression window + `stable_frames` stability requirement; `max_frames_per_segment` hard cap |
| YouTube rate-limiting on many section downloads | Default = one full download; section mode opt-in |
| 8–12 h media fills disk | `--cleanup`; audio-only pass for VAD; frames are JPEG |
| Re-cutting a segment (segment `--force`) invalidates transcribe/slides outputs | Manifest records cut generation per segment; `vsa segment --force` marks dependent stages stale and prints the follow-up commands |

## 16. Open Questions (answer during implementation, not blocking)

1. Pin `asr.language` by default (forcing `"English"` can improve accuracy on English conferences) vs. auto-detect — current default: auto.
2. Whether the opening-slide cue for each segment should fuse in detected segment title text from ASR (nice-to-have).
3. Output container for cuts when the source is a livestream (remux `.mkv`/ts → `.mp4` with `-c copy`; confirm).
