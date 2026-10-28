"""ASR tests: fake, openai_compat (mocked HTTP), qwen_local (stubbed),
chunk planning, cue builder (§13.2)."""

from __future__ import annotations

import json
import math
import sys
import types
import wave
from pathlib import Path

import httpx
import pytest

from video_segment_analysis.asr.base import AsrClient, Utterance, Word
from video_segment_analysis.asr.chunking import plan_chunks
from video_segment_analysis.asr.fake import FakeAsrClient, probe_duration
from video_segment_analysis.asr.openai_compat import OpenAICompatAsrClient
from video_segment_analysis.asr.qwen_local import QwenLocalAsrClient
from video_segment_analysis.config import AsrConfig, Config, SubtitlesConfig
from video_segment_analysis.cues import utterances_to_cues
from video_segment_analysis.errors import ASRError

# ---------------------------------------------------------------------------
# helpers


def make_wav(path: Path, seconds: float, rate: int = 16000) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(int(rate * seconds)):
            frames += int(3000 * math.sin(2 * math.pi * 440 * i / rate)).to_bytes(2, "little", signed=True)
        w.writeframes(bytes(frames))
    return path


def speech_intervals(n: int, period: float = 20.0, dur: float = 10.0) -> list[tuple[float, float]]:
    return [(i * period, i * period + dur) for i in range(n)]


# ---------------------------------------------------------------------------
# fake client


def test_fake_default_spreads_over_duration(tmp_path: Path):
    wav = make_wav(tmp_path / "a.wav", 3.0)
    client = FakeAsrClient()
    utts = client.transcribe(wav)
    assert len(utts) >= 2
    assert utts[0].start >= 0.0
    assert utts[-1].end == pytest.approx(3.0, abs=0.01)
    for a, b in zip(utts, utts[1:]):
        assert b.start >= a.start
        assert all(w.end >= w.start for w in (a.words or []))


def test_fake_canned_scaled_to_duration(tmp_path: Path):
    wav = make_wav(tmp_path / "a.wav", 20.0)
    canned = [Utterance(0.0, 10.0, "one", [Word("one", 0.0, 10.0)]),
              Utterance(5.0, 10.0, "two")]
    utts = FakeAsrClient(canned=canned).transcribe(wav)
    assert [u.end for u in utts] == [pytest.approx(20.0), pytest.approx(20.0)]
    assert [u.start for u in utts] == [pytest.approx(0.0), pytest.approx(10.0)]
    assert utts[0].words[0].end == pytest.approx(20.0)
    assert canned[0].end == 10.0  # originals untouched (copies returned)


def test_fake_fixture_path(tmp_path: Path):
    fx = tmp_path / "canned.json"
    fx.write_text(json.dumps([
        {"start": 0, "end": 1, "text": "from fixture",
         "words": [{"text": "from", "start": 0, "end": 0.5},
                   {"text": "fixture", "start": 0.5, "end": 1}]},
    ]), encoding="utf-8")
    wav = make_wav(tmp_path / "a.wav", 4.0)
    utts = FakeAsrClient(fixture_path=fx).transcribe(wav)
    assert utts[0].text == "from fixture"
    assert utts[0].end == pytest.approx(4.0)
    assert [w.text for w in utts[0].words] == ["from", "fixture"]


def test_probe_duration_wave(tmp_path: Path):
    wav = make_wav(tmp_path / "a.wav", 1.5)
    assert probe_duration(wav) == pytest.approx(1.5, abs=0.01)


# ---------------------------------------------------------------------------
# openai_compat


def openai_client(cfg: AsrConfig, handler) -> OpenAICompatAsrClient:
    return OpenAICompatAsrClient(
        cfg, http_client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_openai_compat_text_only_utterance(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VSA_ASR_API_KEY", raising=False)
    cfg = AsrConfig(openai_base_url="http://srv.test/v1", openai_model="whisper-1")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "Hello world.", "language": "english"})

    wav = make_wav(tmp_path / "c.wav", 2.0)
    utts = openai_client(cfg, handler).transcribe(wav)
    assert len(utts) == 1
    assert utts[0].start == 0.0
    assert utts[0].end == pytest.approx(2.0, abs=0.01)
    assert utts[0].text == "Hello world."
    req = seen[0]
    assert req.url.path == "/v1/audio/transcriptions"
    assert req.headers["content-type"].startswith("multipart/form-data")
    body = req.content
    assert b'name="model"' in body and b"whisper-1" in body
    assert b"c.wav" in body  # file part
    assert "Bearer None" in req.headers.get("authorization", "")  # key unset → placeholder


def test_openai_compat_sends_key_when_env_set(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("VSA_ASR_API_KEY", "sk-secret-123")
    cfg = AsrConfig()
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"text": "ok"})

    openai_client(cfg, handler).transcribe(make_wav(tmp_path / "c.wav", 0.5))
    assert seen[0].headers["authorization"] == "Bearer sk-secret-123"


def test_openai_compat_retries_429_then_success(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VSA_ASR_API_KEY", raising=False)
    cfg = AsrConfig(retries=2, retry_backoff=[0.0, 0.0])
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"text": "recovered"})

    utts = openai_client(cfg, handler).transcribe(make_wav(tmp_path / "c.wav", 1.0))
    assert calls["n"] == 2
    assert utts[0].text == "recovered"


def test_openai_compat_wraps_hard_errors(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("VSA_ASR_API_KEY", raising=False)
    cfg = AsrConfig(retries=0)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": {"message": "boom"}})

    with pytest.raises(ASRError):
        openai_client(cfg, handler).transcribe(make_wav(tmp_path / "c.wav", 0.5))


# ---------------------------------------------------------------------------
# qwen_local (qwen_asr + torch stubbed — never imports the real torch)


class StubItem:
    def __init__(self, text, start_time, end_time):
        self.text, self.start_time, self.end_time = text, start_time, end_time


class StubAlignResult:
    def __init__(self, items):
        self.items = list(items)

    def __iter__(self):
        return iter(self.items)


class StubTranscription:
    def __init__(self, text, time_stamps=None, language="English"):
        self.text, self.time_stamps, self.language = text, time_stamps, language


class StubModel:
    last_kwargs: dict = {}
    last_path: str = ""
    instances: list["StubModel"] = []

    def __init__(self):
        self.calls: list[dict] = []
        StubModel.instances.append(self)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        cls.last_path = pretrained_model_name_or_path
        cls.last_kwargs = kwargs
        return cls()

    def transcribe(self, audio, context="", language=None, return_time_stamps=False):
        self.calls.append({"audio": list(audio), "language": language,
                           "return_time_stamps": return_time_stamps})
        out = []
        for i, _a in enumerate(audio):
            words = StubAlignResult([
                StubItem("Hello", 0.1 + i, 0.4 + i),
                StubItem("world.", 0.5 + i, 0.9 + i),
            ]) if return_time_stamps else None
            out.append(StubTranscription(f"hello world chunk {i}", words))
        return out


@pytest.fixture()
def stubbed_qwen(monkeypatch):
    torch = types.ModuleType("torch")
    for name in ("bfloat16", "float16", "float32"):
        setattr(torch, name, name)
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    torch.version = types.SimpleNamespace(hip="7.2.0", cuda=None)  # type: ignore[attr-defined]
    qwen = types.ModuleType("qwen_asr")
    qwen.Qwen3ASRModel = StubModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", qwen)
    StubModel.instances.clear()
    return qwen


def test_qwen_local_load_kwargs(stubbed_qwen, tmp_path: Path):
    cfg = AsrConfig(model="Qwen/Qwen3-ASR-1.7B", device="cuda:0", dtype="bfloat16",
                    aligner="Qwen/Qwen3-ForcedAligner-0.6B", aligner_device=None,
                    max_new_tokens=2048, max_inference_batch_size=4,
                    attn_implementation="sdpa")
    client = QwenLocalAsrClient(cfg)
    client._load()
    assert StubModel.last_path == cfg.model
    kw = StubModel.last_kwargs
    assert kw["dtype"] == "bfloat16"
    assert kw["device_map"] == "cuda:0"
    assert kw["max_new_tokens"] == 2048
    assert kw["max_inference_batch_size"] == 4
    assert kw["attn_implementation"] == "sdpa"
    assert kw["forced_aligner"] == cfg.aligner
    assert kw["forced_aligner_kwargs"] == {"dtype": "bfloat16", "device_map": "cuda:0"}


def test_qwen_local_maps_utterance_and_words(stubbed_qwen, tmp_path: Path):
    client = QwenLocalAsrClient(AsrConfig())
    wav = make_wav(tmp_path / "c.wav", 5.0)
    utts = client.transcribe(wav)
    model = StubModel.instances[-1]
    assert len(model.calls) == 1
    assert model.calls[0]["return_time_stamps"] is True
    assert len(utts) == 1
    u = utts[0]
    assert u.start == 0.0
    assert u.end == pytest.approx(5.0, abs=0.01)
    assert u.text == "hello world chunk 0"
    assert [w.text for w in u.words] == ["Hello", "world."]
    assert u.words[0].start == pytest.approx(0.1)
    assert u.words[1].end == pytest.approx(0.9)


def test_qwen_local_batch_split(stubbed_qwen, tmp_path: Path):
    client = QwenLocalAsrClient(AsrConfig())
    a = make_wav(tmp_path / "a.wav", 3.0)
    b = make_wav(tmp_path / "b.wav", 7.0)
    res = client.transcribe_batch([a, b])
    model = StubModel.instances[-1]
    assert len(model.calls) == 1  # ONE transcribe call for the batch
    assert len(res) == 2
    assert res[0][0].text == "hello world chunk 0"
    assert res[1][0].text == "hello world chunk 1"
    assert res[0][0].end == pytest.approx(3.0, abs=0.01)
    assert res[1][0].end == pytest.approx(7.0, abs=0.01)


def test_qwen_local_no_aligner_no_timestamps(stubbed_qwen, tmp_path: Path):
    client = QwenLocalAsrClient(AsrConfig(aligner=None))
    utts = client.transcribe(make_wav(tmp_path / "c.wav", 2.0))
    assert StubModel.instances[-1].calls[0]["return_time_stamps"] is False
    assert utts[0].words is None


def test_qwen_local_load_failure_raises_asr_error(monkeypatch, tmp_path: Path):
    torch = types.ModuleType("torch")
    torch.bfloat16 = "bfloat16"  # type: ignore[attr-defined]
    torch.cuda = types.SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
    torch.version = types.SimpleNamespace(hip="7.2.0", cuda=None)  # type: ignore[attr-defined]

    class Boom:
        @classmethod
        def from_pretrained(cls, *a, **k):
            raise RuntimeError("CUDA out of memory")

    qwen = types.ModuleType("qwen_asr")
    qwen.Qwen3ASRModel = Boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "qwen_asr", qwen)
    with pytest.raises(ASRError, match="VRAM"):
        QwenLocalAsrClient(AsrConfig()).transcribe(Path("missing.wav"))


# ---------------------------------------------------------------------------
# chunk planning (§8.3)

CFG = AsrConfig(chunk_target_seconds=120.0, max_chunk_seconds=280.0)


def _assert_plan(intervals, chunks, cap, target):
    assert chunks, "no chunks planned"
    # tiling: contiguous, sorted, first/last pinned to speech edges
    assert chunks[0][0] == intervals[0][0]
    assert chunks[-1][1] == intervals[-1][1]
    for (a_s, a_e), (b_s, b_e) in zip(chunks, chunks[1:]):
        assert a_e == b_s
        assert b_e > b_s
    # cap
    for s, e in chunks:
        assert e - s <= cap + 1e-9
    # boundaries strictly inside silence gaps
    for _, b in chunks[:-1]:
        assert not any(s + 1e-9 < b < e - 1e-9 for s, e in intervals)
    # coverage: every speech interval fully inside a chunk
    for s, e in intervals:
        assert any(cs <= s and e <= ce for cs, ce in chunks)


def test_plan_chunks_basic_tiling():
    ivs = speech_intervals(40)  # 40 × 10 s speech, 10 s gaps, span 790 s
    chunks = plan_chunks(ivs, 0.0, 800.0, CFG)
    _assert_plan(ivs, chunks, cap=CFG.max_chunk_seconds, target=CFG.chunk_target_seconds)
    assert len(chunks) == 3  # ~120 s target over 790 s span


def test_plan_chunks_boundaries_in_gaps():
    ivs = speech_intervals(40)
    chunks = plan_chunks(ivs, 0.0, 800.0, CFG)
    gaps = [(b, a) for (_, b), (a, _) in zip(ivs, ivs[1:])]
    for _, boundary in chunks[:-1]:
        assert any(gs < boundary < ge for gs, ge in gaps)


def test_plan_chunks_prefers_target():
    ivs = speech_intervals(40)
    chunks = plan_chunks(ivs, 0.0, 800.0, CFG)
    for s, e in chunks:  # merged past target → >= target; last chunk may be shorter
        assert e - s >= CFG.chunk_target_seconds - 1e-9


def test_plan_chunks_forced_cap():
    cfg = AsrConfig(chunk_target_seconds=1000.0, max_chunk_seconds=280.0)
    ivs = speech_intervals(30)
    chunks = plan_chunks(ivs, 0.0, 600.0, cfg)
    _assert_plan(ivs, chunks, cap=cfg.max_chunk_seconds, target=cfg.chunk_target_seconds)


def test_plan_chunks_segment_offset_and_clamp():
    ivs = [(10.0, 20.0), (30.0, 40.0), (50.0, 58.0)]
    chunks = plan_chunks(ivs, 10.0, 58.0, AsrConfig(chunk_target_seconds=15.0,
                                                    max_chunk_seconds=60.0))
    _assert_plan(ivs, chunks, cap=60.0, target=15.0)


def test_plan_chunks_edge_cases():
    assert plan_chunks([], 0.0, 100.0, CFG) == []
    big = [(5.0, 400.0)]  # interval longer than cap → own oversized chunk
    assert plan_chunks(big, 0.0, 500.0, CFG) == [(5.0, 400.0)]


def test_plan_chunks_unsorted_input():
    ivs = speech_intervals(5)
    chunks = plan_chunks(list(reversed(ivs)), 0.0, 100.0, CFG)
    _assert_plan(ivs, chunks, cap=CFG.max_chunk_seconds, target=CFG.chunk_target_seconds)


# ---------------------------------------------------------------------------
# cue builder (§8.4)

SUB = SubtitlesConfig(max_cue_duration=10.0, max_cue_chars=42, max_cue_gap=1.5)


def words(text: str, start: float = 0.0, step: float = 0.4) -> list[Word]:
    out = []
    for i, tok in enumerate(text.split()):
        out.append(Word(tok, start + i * step, start + (i + 1) * step - 0.05))
    return out


def test_cues_break_at_sentence_punctuation():
    ws = words("The end. New sentence starts here now.")
    cues = utterances_to_cues([Utterance(0.0, 10.0, "", ws)], cfg=SUB)
    assert len(cues) == 2
    assert cues[0].text == "The end."
    assert cues[0].start == pytest.approx(ws[0].start)
    assert cues[0].end == pytest.approx(ws[1].end)


def test_cues_punctuation_with_closing_quote():
    ws = words('He said "stop!" Then left.')
    cues = utterances_to_cues([Utterance(0.0, 10.0, "", ws)], cfg=SUB)
    assert cues[0].text == 'He said "stop!"'
    assert len(cues) == 2


def test_cues_break_on_max_duration():
    ws = [Word(f"w{i}", i * 2.0, i * 2.0 + 0.5) for i in range(10)]  # 2 s cadence
    cues = utterances_to_cues([Utterance(0.0, 20.0, "", ws)], cfg=SUB)
    assert len(cues) > 1
    for c in cues:
        assert c.end - c.start <= SUB.max_cue_duration + 1e-9


def test_cues_break_on_max_gap():
    ws = words("alpha beta", start=0.0) + words("gamma delta", start=5.0)
    cues = utterances_to_cues([Utterance(0.0, 6.0, "", ws)], cfg=SUB)
    assert len(cues) == 2
    assert cues[1].start == pytest.approx(5.0)


def test_cues_break_on_max_chars():
    ws = words(" ".join(["word"] * 20))  # ~100 chars, short durations
    cues = utterances_to_cues([Utterance(0.0, 10.0, "", ws)], cfg=SUB)
    assert len(cues) >= 3
    for c in cues:
        assert len(c.text) <= SUB.max_cue_chars + len("word")  # word-wrap keeps words


def test_cues_cjk_tokens_joined_without_spaces():
    ws = [Word("你", 0.0, 0.2), Word("好", 0.2, 0.4), Word("世界", 0.4, 0.8)]
    cues = utterances_to_cues([Utterance(0.0, 1.0, "你好世界", ws)], cfg=SUB)
    assert cues[-1].text == "你好世界"


def test_cues_no_words_single_short_utterance():
    cues = utterances_to_cues([Utterance(2.0, 8.0, "Just one short sentence.")], cfg=SUB)
    assert len(cues) == 1
    assert (cues[0].start, cues[0].end) == (2.0, 8.0)


def test_cues_no_words_proportional_split():
    text = "First sentence here. Second one follows. Third!"
    cues = utterances_to_cues([Utterance(10.0, 60.0, text)], cfg=SUB)
    assert len(cues) == 3
    # window fully covered, monotonic, non-overlapping
    assert cues[0].start == pytest.approx(10.0)
    assert cues[-1].end == pytest.approx(60.0)
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start + 1e-9
        assert b.start > a.start
    # proportional by char count
    total = sum(len(c.text) for c in cues)
    assert cues[0].end - cues[0].start == pytest.approx(
        50.0 * len(cues[0].text) / total, rel=0.1)


def test_cues_no_words_long_sentence_wrapped():
    long_text = "word " * 40  # no sentence end within max_cue_chars
    cues = utterances_to_cues([Utterance(0.0, 100.0, long_text)], cfg=SUB)
    assert len(cues) >= 2
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start + 1e-9
    assert cues[-1].end == pytest.approx(100.0)


def test_cues_integration_normalize_gap_and_clamp():
    # word cue touching the next utterance's cue → gap enforced
    ws = [Word("one.", 0.0, 5.0)]
    utts = [Utterance(0.0, 5.0, "", ws), Utterance(5.0, 9.0, "Second sentence.")]
    cues = utterances_to_cues(utts, cfg=SUB, segment_duration=9.0)
    for a, b in zip(cues, cues[1:]):
        assert a.end <= b.start - 0.02 + 1e-9
    assert cues[-1].end <= 9.0
    assert [c.index for c in cues] == list(range(1, len(cues) + 1))


def test_cues_empty_text_skipped():
    assert utterances_to_cues([Utterance(0.0, 5.0, "   ")], cfg=SUB) == []


# ---------------------------------------------------------------------------
# dispatch


def test_build_asr_client_backends(tmp_path: Path):
    from video_segment_analysis.asr import build_asr_client
    cfg = Config()
    cfg.asr.backend = "fake"
    assert isinstance(build_asr_client(cfg), FakeAsrClient)
    cfg = Config()
    cfg.asr.backend = "openai_compat"
    assert isinstance(build_asr_client(cfg), OpenAICompatAsrClient)
    cfg.asr.backend = "qwen_local"
    client = build_asr_client(cfg)
    assert isinstance(client, QwenLocalAsrClient)
    assert isinstance(client, AsrClient)
