# vsa — Long-Video Segment Analysis

Split long conference / livestream recordings into per-talk segments,
transcribe each segment with Qwen3-ASR (word-level timestamps), and describe
slide changes with a vision-language model — producing per-segment media,
JSON transcripts, SRT/VTT subtitles, and slide-timestamped descriptions.

Design: see [DESIGN.md](DESIGN.md).

## Requirements

- [uv](https://docs.astral.sh/uv/) (all Python runs through `uv run` — no
  venv activation, no bare `pip`/`python`)
- `ffmpeg` + `ffprobe` on `PATH` (system binaries, invoked via subprocess)
- An OpenAI-compatible ASR endpoint serving **Qwen3-ASR** for transcription
  (local GPU mode: `asr.backend = "qwen-local"`)
- Any OpenAI-compatible VLM with vision for the `slides` stage

## Install

```bash
uv sync            # installs deps from uv.lock, registers the `vsa` CLI
uv run vsa --help
```

## Quick start

```bash
# 1) Segment: audio extraction → silence detection → segment planning → cuts
uv run vsa segment --source https://www.youtube.com/watch?v=XXXXXXXXXXX

# 2) Transcribe each segment (needs ASR endpoint, see config below)
uv run vsa transcribe

# 3) Slide-change detection + VLM descriptions (needs VLM)
uv run vsa slides

# All of the above in one go
uv run vsa all --source <url-or-path>
```

`--source` accepts a local media file or any yt-dlp-supported URL. All
stages are resumable: rerun the same command to continue; the manifest
(`out/<project>/manifest.json`) is the single source of truth for progress.

Useful flags:

| Flag | Effect |
| --- | --- |
| `--out DIR` | output directory (default from config `paths.workdir`) |
| `--config FILE` | YAML config (defaults + `config.example.yaml`) |
| `--segment-id ID` | restrict a run to specific segments (repeatable) |
| `--force` | redo stages; with `--segment-id` only the cut stage is forced |
| `--strategy` | `auto\|local-copy\|local-encode\|yt-sections` (segment) |
| `--dry-run` | print the segment plan without touching media |

## Output layout

```
out/<project>/
├── manifest.json          # canonical state: source, vad, segments, asr, slides
├── audio.wav              # one full 16 kHz mono decode (VAD + ASR source)
├── source_video.mp4       # full video (YouTube auto-strategy downloads once)
├── segments/seg_NNN.mp4   # per-segment video cut
├── segments/seg_NNN.wav   # per-segment audio cut (sample-accurate)
├── transcripts/seg_NNN.json   # word-level timestamps
├── subtitles/seg_NNN.srt/.vtt
└── slides/seg_NNN/            # slide frames + descriptions
```

## Configuration

Copy `config.example.yaml`, edit, and pass `--config`. Secrets are **never**
stored in config — set the env var named by `*_api_key_env`:

```bash
export VSA_ASR_API_KEY=...    # asr.api_key_env  (default VSA_ASR_API_KEY)
export VSA_VLM_API_KEY=...    # vlm.api_key_env  (default VSA_VLM_API_KEY)
```

Any setting can be overridden via environment:
`VSA__<SECTION>__<KEY>` (e.g. `VSA__VAD__NOISE_DB=-40`).

Key segmentation knobs (`segmentation` section):

- `min_silence_gap` — silence length that ends a segment (default 45 s)
- `min_segment_duration` — shorter speech runs are dropped (default 120 s)
- `pad_start` / `pad_end` — padding around speech (default 3 s / 5 s)

## Cutting notes (YouTube)

The default `auto` strategy downloads the full video **once**, then cuts
every segment locally with `-c copy` (seconds per segment). Stream-copy
cuts can start on an earlier keyframe; when the measured drift exceeds 3 s
the manifest marks the segment `reencode_recommended` and prints a repair
hint:

```bash
uv run vsa segment --source <url> --strategy local-encode \
  --segment-id seg_005 --force
```

Audio for ASR always comes from the single full WAV, so transcript timing
is unaffected by video-cut drift.

## Development

```bash
uv run pytest tests/ -q          # unit tests (network tests excluded)
uv run pytest tests/ -m network  # opt-in: real yt-dlp metadata checks
```
