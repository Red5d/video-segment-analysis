"""qwen_local device preflight (§8.2 diagnostics, AMD/ROCm aware)."""

from __future__ import annotations

import sys
import types

import pytest

from video_segment_analysis.asr.qwen_local import _preflight_device
from video_segment_analysis.config import AsrConfig
from video_segment_analysis.errors import ASRError


def _fake_torch(*, hip: str | None, available: bool) -> types.ModuleType:
    m = types.ModuleType("torch")
    m.cuda = types.SimpleNamespace(is_available=lambda: available)
    m.version = types.SimpleNamespace(hip=hip, cuda=None if hip else "13.0")
    return m


def test_preflight_cpu_needs_no_torch():
    # device=cpu must work in a torch-less environment (lazy import skipped)
    _preflight_device(AsrConfig(device="cpu"))


def test_preflight_cuda_build_without_nvidia(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hip=None, available=False))
    with pytest.raises(ASRError, match="AMD GPU.*ROCm|ROCm.*AMD"):
        _preflight_device(AsrConfig())


def test_preflight_rocm_build_without_visible_gpu(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hip="7.2.4", available=False))
    with pytest.raises(ASRError, match="ROCm build"):
        _preflight_device(AsrConfig())


def test_preflight_ok_when_gpu_available(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(hip="7.2.4", available=True))
    _preflight_device(AsrConfig())
