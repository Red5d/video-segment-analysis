"""OpenAI-compatible transcription backend (DESIGN §8.2).

``POST {base_url}/audio/transcriptions`` (multipart WAV + model) via the
``openai`` SDK. The endpoint returns plain text → one coarse
``Utterance(0, audio_duration, text)``; the cue builder applies the
no-word proportional timing path. The API key is read from
``cfg.openai_api_key_env`` at construction and never logged.
"""

from __future__ import annotations

import contextlib
import logging
import os
import wave
from pathlib import Path
from typing import Sequence

import httpx
import openai

from ..config import AsrConfig
from ..errors import ASRError
from ..retry import retry_with_backoff
from .base import AsrClient, Utterance

logger = logging.getLogger("vsa.asr.openai_compat")


def _wav_duration(path: Path) -> float:
    with contextlib.closing(wave.open(str(path))) as w:
        rate = w.getframerate()
        return w.getnframes() / rate if rate else 0.0


class OpenAICompatAsrClient(AsrClient):
    def __init__(self, cfg: AsrConfig, *, http_client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        key = os.environ.get(cfg.openai_api_key_env) or "None"  # local servers need no key
        self._client = openai.OpenAI(
            base_url=cfg.openai_base_url,
            api_key=key,
            timeout=cfg.request_timeout,
            max_retries=0,  # retry_with_backoff owns retries
            **({"http_client": http_client} if http_client is not None else {}),
        )

    def transcribe(self, audio_path: Path, language: str | None = None) -> list[Utterance]:
        path = Path(audio_path)
        logger.debug("transcribing %s via %s (model=%s)",
                     path.name, self.cfg.openai_base_url, self.cfg.openai_model)

        def call() -> object:
            kwargs = {"file": open(path, "rb")}
            try:
                return self._client.audio.transcriptions.create(
                    model=self.cfg.openai_model,
                    language=language,
                    **kwargs,
                )
            finally:
                kwargs["file"].close()

        try:
            result = retry_with_backoff(
                call,
                retries=self.cfg.retries,
                backoff=self.cfg.retry_backoff,
                retry_on=(openai.RateLimitError, openai.APITimeoutError,
                          openai.APIConnectionError),
                label=f"asr.openai_compat {path.name}",
            )
        except openai.OpenAIError as exc:
            raise ASRError(f"openai_compat transcription failed for {path}: {exc}") from exc

        text = getattr(result, "text", result)
        text = str(text or "").strip()
        if not text:
            return []
        return [Utterance(0.0, _wav_duration(path), text)]

    def transcribe_batch(self, audio_paths: Sequence[Path],
                         language: str | None = None) -> list[list[Utterance]]:
        """Sequential per-chunk transcription (the pipeline parallelizes via
        ``cfg.concurrency``)."""
        return [self.transcribe(p, language) for p in audio_paths]
