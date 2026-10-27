"""VLM client contract tests against a mocked OpenAI-compatible endpoint.

The OpenAI SDK speaks httpx; the ``responses`` library mocks ``requests``.
A thin httpx transport backed by ``requests`` bridges the two so every SDK
call is intercepted offline (no network, no ffmpeg).
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import cv2
import httpx
import numpy as np
import pytest
import requests
import responses

from video_segment_analysis.config import Config, VlmConfig
from video_segment_analysis.errors import VLMError
from video_segment_analysis.vlm import FakeVlmClient, build_vlm_client
from video_segment_analysis.vlm.base import UNAVAILABLE_TEXT
from video_segment_analysis.vlm.openai_compat import (
    DEFAULT_PROMPT,
    OpenAICompatVlmClient,
)

BASE_URL = "http://vsa.test/v1"
ENDPOINT = BASE_URL + "/chat/completions"
API_KEY = "sk-TEST-SECRET-0123456789abcdef"


class RequestsTransport(httpx.BaseTransport):
    """Route httpx requests through ``requests`` so ``responses`` sees them."""

    def __init__(self) -> None:
        self._session = requests.Session()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length", "transfer-encoding")
        }
        resp = self._session.request(
            request.method,
            str(request.url),
            headers=headers,
            data=request.content,
            allow_redirects=False,
        )
        out_headers = [
            (k, v) for k, v in resp.headers.items() if k.lower() != "transfer-encoding"
        ]
        return httpx.Response(
            resp.status_code, headers=out_headers, content=resp.content, request=request
        )


def vlm_cfg(**overrides) -> VlmConfig:
    base = dict(
        base_url=BASE_URL,
        model="test-vision-model",
        api_key_env="VSA_VLM_API_KEY",
        retries=2,
        retry_backoff=[0.0, 0.0],
        request_timeout=5,
    )
    base.update(overrides)
    return VlmConfig(**base)


def make_client(cfg: VlmConfig) -> OpenAICompatVlmClient:
    return OpenAICompatVlmClient(cfg, http_client=httpx.Client(transport=RequestsTransport()))


def completion_json(text: str) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1730000000,
        "model": "test-vision-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


@pytest.fixture()
def jpeg(tmp_path: Path) -> Path:
    img = np.zeros((24, 32, 3), dtype=np.uint8)
    img[4:20, 4:28] = (40, 120, 220)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    path = tmp_path / "slide_002.jpg"
    path.write_bytes(buf.tobytes())
    return path


# -- request shape -----------------------------------------------------------


@responses.activate
def test_image_data_url_and_prompt_sent(jpeg: Path):
    responses.add(
        responses.POST, ENDPOINT,
        json=completion_json('  "Slide 1 title\nwith two lines"  '), status=200,
    )
    client = make_client(vlm_cfg())
    text = client.describe(jpeg)

    assert text == "Slide 1 title with two lines"  # whitespace collapsed, quotes stripped
    body = json.loads(responses.calls[0].request.body)
    assert body["model"] == "test-vision-model"
    content = body["messages"][0]["content"]
    image_parts = [p for p in content if p.get("type") == "image_url"]
    text_parts = [p for p in content if p.get("type") == "text"]
    assert len(image_parts) == 1 and len(text_parts) == 1
    url = image_parts[0]["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    assert base64.b64decode(url.split(",", 1)[1]).startswith(b"\xff\xd8\xff")  # JPEG magic
    assert text_parts[0]["text"] == DEFAULT_PROMPT
    assert "transcribe all visible text verbatim" in text_parts[0]["text"]


@responses.activate
def test_prompt_file_override(jpeg: Path, tmp_path: Path):
    prompt = "Custom slide prompt."
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text(prompt + "\n", encoding="utf-8")
    responses.add(responses.POST, ENDPOINT, json=completion_json("ok"), status=200)
    make_client(vlm_cfg(prompt_file=str(prompt_file))).describe(jpeg)
    body = json.loads(responses.calls[0].request.body)
    sent = [p["text"] for p in body["messages"][0]["content"] if p.get("type") == "text"]
    assert sent == [prompt]


# -- retries / failures ------------------------------------------------------


@responses.activate
def test_retry_on_429_then_success(jpeg: Path):
    responses.add(responses.POST, ENDPOINT, json={"error": {"message": "slow down"}}, status=429)
    responses.add(responses.POST, ENDPOINT, json=completion_json("Recovered slide text"), status=200)
    result = make_client(vlm_cfg()).describe_checked(jpeg)
    assert result.failed is False
    assert result.text == "Recovered slide text"
    assert len(responses.calls) == 2


@responses.activate
def test_gives_up_with_placeholder(jpeg: Path):
    responses.add(responses.POST, ENDPOINT, json={"error": {"message": "boom"}}, status=500)
    responses.add(responses.POST, ENDPOINT, json={"error": {"message": "boom"}}, status=500)
    result = make_client(vlm_cfg(retries=1)).describe_checked(jpeg)
    assert result.failed is True
    assert result.text == UNAVAILABLE_TEXT
    assert len(responses.calls) == 2


@responses.activate
def test_api_key_redacted_in_logs(jpeg: Path, monkeypatch, caplog):
    monkeypatch.setenv("VSA_VLM_API_KEY", API_KEY)
    responses.add(responses.POST, ENDPOINT, json={"error": {"message": "boom"}}, status=500)
    responses.add(responses.POST, ENDPOINT, json={"error": {"message": "boom"}}, status=500)
    caplog.set_level(logging.DEBUG, logger="vsa")
    caplog.set_level(logging.WARNING, logger="openai")
    result = make_client(vlm_cfg(retries=1)).describe_checked(jpeg)
    assert result.failed is True
    assert caplog.records  # we did log the retries/failure
    for record in caplog.records:
        assert API_KEY not in record.getMessage()
        assert API_KEY not in caplog.text


# -- startup validation ------------------------------------------------------


def test_hosted_base_url_missing_key_raises(monkeypatch):
    monkeypatch.delenv("VSA_VLM_API_KEY", raising=False)
    with pytest.raises(VLMError, match="VSA_VLM_API_KEY"):
        OpenAICompatVlmClient(VlmConfig())  # default base_url = api.openai.com
    with pytest.raises(VLMError, match="VSA_VLM_API_KEY"):
        OpenAICompatVlmClient(VlmConfig(base_url="https://openrouter.ai/api/v1"))
    with pytest.raises(VLMError, match="MY_CUSTOM_KEY"):
        OpenAICompatVlmClient(VlmConfig(api_key_env="MY_CUSTOM_KEY"))


def test_local_base_url_without_key_is_ok(monkeypatch):
    monkeypatch.delenv("VSA_VLM_API_KEY", raising=False)
    client = OpenAICompatVlmClient(VlmConfig(base_url="http://127.0.0.1:11434/v1"))
    assert client.name.startswith("openai_compat:")


# -- fake backend / factory ----------------------------------------------------


def test_fake_client_is_deterministic(tmp_path: Path):
    client = FakeVlmClient()
    a, b = tmp_path / "slide_001.jpg", tmp_path / "slide_002.jpg"
    first = client.describe(a)
    assert first == client.describe(a)  # stable per path
    assert first == client.describe(a)
    assert client.describe(b) != first
    assert "slide_001" in first and "fixture #" in first
    checked = client.describe_checked(b)
    assert checked.failed is False and checked.text == client.describe(b)


def test_build_vlm_client_factory(monkeypatch):
    monkeypatch.setenv("VSA_VLM_API_KEY", API_KEY)
    cfg = Config()  # default backend openai_compat (hosted base_url → key required)
    assert isinstance(build_vlm_client(cfg), OpenAICompatVlmClient)
    cfg.vlm.backend = "fake"
    assert isinstance(build_vlm_client(cfg), FakeVlmClient)
