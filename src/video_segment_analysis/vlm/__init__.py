"""VLM backends and backend factory."""

from __future__ import annotations

from ..config import Config
from ..errors import ConfigError
from .base import UNAVAILABLE_TEXT, SlideDescription, VlmClient
from .fake import FakeVlmClient

__all__ = [
    "VlmClient",
    "SlideDescription",
    "UNAVAILABLE_TEXT",
    "FakeVlmClient",
    "OpenAICompatVlmClient",
    "build_vlm_client",
]


def build_vlm_client(cfg: Config) -> VlmClient:
    """Instantiate the configured VLM backend (``vlm.backend``)."""
    backend = cfg.vlm.backend
    if backend == "fake":
        return FakeVlmClient()
    if backend == "openai_compat":
        from .openai_compat import OpenAICompatVlmClient  # lazy: pulls the SDK

        return OpenAICompatVlmClient(cfg.vlm)
    raise ConfigError(f"unknown vlm backend: {backend!r}")
