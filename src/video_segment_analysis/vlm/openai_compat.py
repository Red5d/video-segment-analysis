"""Generic OpenAI-compatible VLM client (DESIGN §9.4).

``chat.completions.create`` with one base64-JPEG ``image_url`` part plus the
prompt. API key comes from ``vlm.api_key_env`` (omitted for local servers);
a hosted base_url with the env var unset is a startup error (§11). Transient
failures (429/5xx/timeout/connection) retry via ``retry_with_backoff``; final
failure yields the ``[slide content unavailable]`` placeholder flagged
``vlm_failed``. The API key is redacted from every log line.
"""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np

from ..config import VlmConfig
from ..errors import VLMError
from ..retry import retry_with_backoff
from .base import UNAVAILABLE_TEXT, SlideDescription, VlmClient

log = logging.getLogger("vsa.vlm")

#: Built-in default prompt, DESIGN §9.4 verbatim.
DEFAULT_PROMPT = (
    "You are given one frame from a conference presentation, captured at the "
    "moment the slide changed. Describe the slide content for someone who "
    "cannot see it: transcribe all visible text verbatim (title first, then "
    "bullets in order), describe charts/diagrams and what they show, and "
    "mention speaker name/affiliation if visible on the slide. Output plain "
    "text, no markdown, no commentary about the image quality. Keep under "
    "250 words."
)


def _requires_key(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").lower()
    return host == "api.openai.com" or host.endswith(".openai.com") or "openrouter" in host


def _clean(text: str) -> str:
    """Collapse whitespace, strip surrounding quotes."""
    text = " ".join(text.split())
    for quote in ('"', "'", "“", "”"):
        text = text.strip(quote)
    return text.strip()


class _TransientAPIError(Exception):
    """A retryable HTTP status (429/5xx), key-redacted."""


class OpenAICompatVlmClient(VlmClient):
    def __init__(self, cfg: VlmConfig, *, http_client: object | None = None) -> None:
        from openai import OpenAI  # lazy import: keeps `fake` usable without the SDK

        self._cfg = cfg
        key = os.environ.get(cfg.api_key_env)
        if _requires_key(cfg.base_url) and not key:
            raise VLMError(
                f"VLM base_url {cfg.base_url!r} requires an API key; "
                f"set the {cfg.api_key_env} environment variable "
                f"(or point vlm.base_url at a local server)"
            )
        self._key = key
        self._prompt = self._load_prompt(cfg)
        self._client = OpenAI(
            base_url=cfg.base_url,
            api_key=key or "not-needed",  # local servers ignore it; header is still sent
            timeout=cfg.request_timeout,
            max_retries=0,  # retries belong to retry_with_backoff
            **({"http_client": http_client} if http_client is not None else {}),
        )

    @property
    def name(self) -> str:
        return f"openai_compat:{self._cfg.model}"

    @staticmethod
    def _load_prompt(cfg: VlmConfig) -> str:
        if cfg.prompt_file:
            path = Path(cfg.prompt_file)
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise VLMError(f"cannot read vlm.prompt_file {path}: {exc}") from exc
        return DEFAULT_PROMPT

    # -- request path -------------------------------------------------------

    def _redact(self, message: str) -> str:
        return message.replace(self._key, "***redacted***") if self._key else message

    def _image_data_url(self, image_path: Path) -> str:
        raw = image_path.read_bytes()
        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is not None:  # re-encode to honour image_quality
            ok, buf = cv2.imencode(
                ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self._cfg.image_quality]
            )
            if ok:
                raw = buf.tobytes()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")

    def describe_checked(self, image_path: Path) -> SlideDescription:
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        data_url = self._image_data_url(image_path)
        label = f"vlm:{image_path.name}"

        def _call() -> str:
            try:
                resp = self._client.chat.completions.create(
                    model=self._cfg.model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": data_url}},
                                {"type": "text", "text": self._prompt},
                            ],
                        }
                    ],
                    timeout=self._cfg.request_timeout,
                )
            except APIStatusError as exc:  # incl. RateLimitError
                if exc.status_code == 429 or exc.status_code >= 500:
                    raise _TransientAPIError(
                        self._redact(f"HTTP {exc.status_code}: {exc}")
                    ) from exc
                raise
            text = (resp.choices[0].message.content or "").strip()
            if not text:
                raise _TransientAPIError("empty completion")
            return _clean(text)

        try:
            text = retry_with_backoff(
                _call,
                retries=self._cfg.retries,
                backoff=self._cfg.retry_backoff,
                retry_on=(_TransientAPIError, APIConnectionError, APITimeoutError),
                label=label,
            )
        except Exception as exc:
            log.error("%s: giving up after retries: %s", label, self._redact(str(exc)))
            return SlideDescription(UNAVAILABLE_TEXT, failed=True)
        return SlideDescription(text)

    def describe(self, image_path: Path) -> str:
        return self.describe_checked(image_path).text
