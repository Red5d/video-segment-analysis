"""ASR backend dispatch (DESIGN §8.2)."""

from __future__ import annotations

from ..config import Config
from ..errors import ConfigError
from .base import AsrClient, Utterance, Word
from .fake import FakeAsrClient
from .openai_compat import OpenAICompatAsrClient
from .qwen_local import QwenLocalAsrClient

__all__ = ["AsrClient", "Utterance", "Word", "build_asr_client",
           "FakeAsrClient", "OpenAICompatAsrClient", "QwenLocalAsrClient"]


def build_asr_client(cfg: Config) -> AsrClient:
    """Instantiate the client for ``cfg.asr.backend``."""
    backend = cfg.asr.backend
    if backend == "fake":
        return FakeAsrClient()
    if backend == "openai_compat":
        return OpenAICompatAsrClient(cfg.asr)
    if backend == "qwen_local":
        return QwenLocalAsrClient(cfg.asr)
    raise ConfigError(f"unknown asr backend: {backend!r}")
