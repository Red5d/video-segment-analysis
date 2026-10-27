"""Configuration: pydantic models with defaults matching config.example.yaml.

Load order (later wins): built-in defaults → YAML file → env vars
(``VSA__<SECTION>__<KEY>``) → explicit CLI overrides passed to
:func:`load_config`. API keys are never stored here; only the *names* of
env vars live in the config, and clients read the values from the
environment at construction time.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

from .errors import ConfigError

SourceType = Literal["auto", "local", "youtube"]
CutStrategy = Literal["auto", "local-copy", "local-encode", "yt-sections"]
VadEngine = Literal["ffmpeg", "silero"]
AsrBackend = Literal["qwen_local", "openai_compat", "fake"]
VlmBackend = Literal["openai_compat", "fake"]


class SourceConfig(BaseModel):
    type: SourceType = "auto"
    cookiefile: str | None = None
    video_format: str = "bv*[height<=1080]+ba/b"


class PathsConfig(BaseModel):
    workdir: Path = Path("./out")


class AudioConfig(BaseModel):
    sample_rate: int = 16000
    channels: int = 1
    format: Literal["wav"] = "wav"


class VadConfig(BaseModel):
    engine: VadEngine = "ffmpeg"
    noise_db: int = -35
    min_silence_duration: float = 2.0


class SegmentationConfig(BaseModel):
    min_silence_gap: float = 45.0
    min_segment_duration: float = 120.0
    pad_start: float = 3.0
    pad_end: float = 5.0
    merge_speech_gap: float = 2.0


class CutConfig(BaseModel):
    strategy: CutStrategy = "auto"
    video_format_local: str = "copy"


class AsrConfig(BaseModel):
    backend: AsrBackend = "qwen_local"
    # --- qwen_local ---
    model: str = "Qwen/Qwen3-ASR-1.7B"
    device: str = "cuda:0"
    dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    aligner: str | None = "Qwen/Qwen3-ForcedAligner-0.6B"
    aligner_device: str | None = None
    language: str | None = None
    max_new_tokens: int = 1024
    max_inference_batch_size: int = 8
    attn_implementation: str = "sdpa"
    # --- chunking (all backends) ---
    chunk_target_seconds: float = 120.0
    max_chunk_seconds: float = 280.0
    # --- openai_compat ---
    openai_base_url: str = "http://localhost:8000/v1"
    openai_model: str = "whisper-1"
    openai_api_key_env: str = "VSA_ASR_API_KEY"
    # --- shared request behavior ---
    concurrency: int = 1
    retries: int = 2
    retry_backoff: list[float] = Field(default_factory=lambda: [2, 8, 30])
    request_timeout: int = 600

    @property
    def aligner_device_resolved(self) -> str:
        return self.aligner_device or self.device


class SubtitlesConfig(BaseModel):
    formats: list[Literal["srt", "vtt"]] = Field(default_factory=lambda: ["srt", "vtt"])
    max_cue_duration: float = 10.0
    max_cue_chars: int = 42
    max_cue_gap: float = 1.5


class SlidesConfig(BaseModel):
    sample_fps: float = 1.0
    analyze_width: int = 320
    pixel_diff_threshold: int = 28
    change_ratio: float = 0.5
    min_change_interval: float = 3.0
    stable_frames: int = 2
    frame_offset: float = 1.5
    max_frames_per_segment: int = 200
    crop: str | None = None  # "x,y,w,h"

    @field_validator("crop")
    @classmethod
    def _check_crop(cls, v: str | None) -> str | None:
        if v is None:
            return v
        parts = [p.strip() for p in v.split(",")]
        if len(parts) != 4:
            raise ValueError("crop must be 'x,y,w,h'")
        ints = [int(p) for p in parts]
        if ints[2] <= 0 or ints[3] <= 0:
            raise ValueError("crop width/height must be positive")
        return ",".join(str(i) for i in ints)

    @property
    def crop_tuple(self) -> tuple[int, int, int, int] | None:
        if not self.crop:
            return None
        x, y, w, h = (int(p) for p in self.crop.split(","))
        return x, y, w, h


class VlmConfig(BaseModel):
    backend: VlmBackend = "openai_compat"
    model: str = "gpt-4o"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "VSA_VLM_API_KEY"
    image_format: Literal["jpeg"] = "jpeg"
    image_quality: int = 85
    prompt_file: str | None = None
    concurrency: int = 2
    retries: int = 3
    retry_backoff: list[float] = Field(default_factory=lambda: [2, 8, 30])
    request_timeout: int = 120


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str | None = None


class Config(BaseModel):
    source: SourceConfig = Field(default_factory=SourceConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    cut: CutConfig = Field(default_factory=CutConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    subtitles: SubtitlesConfig = Field(default_factory=SubtitlesConfig)
    slides: SlidesConfig = Field(default_factory=SlidesConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _env_overrides(env: dict[str, str] | None = None) -> dict:
    """Collect ``VSA__SECTION__KEY`` environment variables into a nested dict."""
    env = os.environ if env is None else env
    result: dict[str, Any] = {}
    for name, raw in env.items():
        if not name.startswith("VSA__"):
            continue
        parts = name[len("VSA__"):].lower().split("__")
        if len(parts) != 2:
            continue
        section, key = parts
        node = result.setdefault(section, {})
        try:
            node[key] = yaml.safe_load(raw)  # typed parse (int/float/bool/list)
        except yaml.YAMLError:
            node[key] = raw
    return result


def load_config(
    path: str | Path | None = None,
    *,
    overrides: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """Build a :class:`Config` from defaults + optional YAML + env + overrides.

    ``overrides`` is a nested dict keyed by section (from CLI flags); it has
    the highest precedence.
    """
    data: dict[str, Any] = {}
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise ConfigError(f"config file not found: {p}")
        loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ConfigError(f"config file must be a YAML mapping: {p}")
        data = _deep_merge(data, loaded)
    data = _deep_merge(data, _env_overrides(env))
    if overrides:
        data = _deep_merge(data, overrides)
    try:
        return Config.model_validate(data)
    except Exception as exc:  # pydantic ValidationError
        raise ConfigError(f"invalid configuration: {exc}") from exc
