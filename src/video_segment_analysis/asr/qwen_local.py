"""Local Qwen3-ASR backend via the official ``qwen_asr`` package (DESIGN §8.2).

``torch`` and ``qwen_asr`` are imported lazily inside :meth:`_load` so the
module imports fine (and is testable) without torch. One model instance is
built per client and reused for every chunk; the pipeline batches chunks
through :meth:`QwenLocalAsrClient.transcribe_batch`.

Real installed API (qwen-asr 0.0.6):
  Qwen3ASRModel.from_pretrained(path, *, forced_aligner=None,
      forced_aligner_kwargs=None, max_inference_batch_size=32,
      max_new_tokens=512, **hf_kwargs)                      # hf: dtype, device_map, ...
  model.transcribe(audio=str|list[str]|(ndarray,sr)|list, context="",
      language=None, return_time_stamps=False) -> list[ASRTranscription]
  ASRTranscription(language: str, text: str,
      time_stamps: ForcedAlignResult | None)
  ForcedAlignResult: iterable; .items: list[ForcedAlignItem]
  ForcedAlignItem(text: str, start_time: float, end_time: float)   # seconds
"""

from __future__ import annotations

import contextlib
import logging
import threading
import wave
from pathlib import Path
from typing import Sequence

from ..config import AsrConfig
from ..errors import ASRError
from .base import AsrClient, Utterance, Word

logger = logging.getLogger("vsa.asr.qwen_local")

_LOAD_GUIDANCE = (
    "Failed to load Qwen3-ASR model. Checks:\n"
    "  * VRAM: 1.7B bf16 needs ~5 GB (+~1.5 GB aligner); use the 0.6B model\n"
    "    (asr.model: Qwen/Qwen3-ASR-0.6B), dtype: float16, and\n"
    "    max_inference_batch_size: 1-2 on small GPUs.\n"
    "  * No GPU? Set asr.device: cpu (works, slow) and check asr.device points\n"
    "    at a valid CUDA index.\n"
    "  * First use downloads weights from Hugging Face — check network access\n"
    "    and HF cache permissions, or pre-download with `huggingface-cli download`."
)


def _wav_duration(path: Path) -> float:
    with contextlib.suppress(Exception):
        with wave.open(str(path)) as w:
            rate = w.getframerate()
            if rate:
                return w.getnframes() / rate
    return 0.0


def _map_time_stamps(ts: object) -> list[Word] | None:
    """ForcedAlignResult (or any iterable of .text/.start_time/.end_time
    items, incl. dicts) → list[Word]."""
    if ts is None:
        return None
    items = getattr(ts, "items", ts)
    words: list[Word] = []
    try:
        iterator = iter(items)  # type: ignore[call-overload]
    except TypeError:
        return None
    for it in iterator:
        if isinstance(it, dict):
            text, s, e = it.get("text", ""), it.get("start_time", 0), it.get("end_time", 0)
        else:
            text = getattr(it, "text", "")
            s = getattr(it, "start_time", getattr(it, "start", 0.0))
            e = getattr(it, "end_time", getattr(it, "end", 0.0))
        if str(text).strip():
            words.append(Word(str(text), float(s), float(e)))
    return words


class QwenLocalAsrClient(AsrClient):
    """Transformers-backend Qwen3-ASR with optional forced aligner."""

    def __init__(self, cfg: AsrConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._lock = threading.Lock()

    # -- model lifecycle ----------------------------------------------------

    def _load(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            try:
                import torch
                from qwen_asr import Qwen3ASRModel

                dtype = getattr(torch, self.cfg.dtype)  # bfloat16|float16|float32
                kwargs: dict = {
                    "dtype": dtype,
                    "device_map": self.cfg.device,
                    "max_inference_batch_size": self.cfg.max_inference_batch_size,
                    "max_new_tokens": self.cfg.max_new_tokens,
                    "attn_implementation": self.cfg.attn_implementation,
                }
                if self.cfg.aligner:
                    kwargs["forced_aligner"] = self.cfg.aligner
                    kwargs["forced_aligner_kwargs"] = {
                        "dtype": dtype,
                        "device_map": self.cfg.aligner_device_resolved,
                    }
                logger.info("loading Qwen3-ASR %s (device=%s dtype=%s aligner=%s)",
                            self.cfg.model, self.cfg.device, self.cfg.dtype,
                            self.cfg.aligner)
                if self.cfg.device == "cpu":
                    logger.warning("asr.device=cpu: transcription will be very slow; "
                                   "a GPU is strongly recommended")
                self._model = Qwen3ASRModel.from_pretrained(self.cfg.model, **kwargs)
            except (ImportError, RuntimeError, OSError, ValueError) as exc:
                raise ASRError(f"{_LOAD_GUIDANCE}\n\nUnderlying error: {exc}") from exc
            return self._model

    # -- transcription --------------------------------------------------------

    def transcribe(self, audio_path: Path, language: str | None = None) -> list[Utterance]:
        return self.transcribe_batch([Path(audio_path)], language)[0]

    def transcribe_batch(self, audio_paths: Sequence[Path],
                         language: str | None = None) -> list[list[Utterance]]:
        """One ``model.transcribe`` call for all chunks; results are split
        back per input (qwen_asr returns one ASRTranscription per audio)."""
        model = self._load()
        paths = [Path(p) for p in audio_paths]
        if not paths:
            return []
        lang = language or self.cfg.language
        want_ts = bool(self.cfg.aligner)
        try:
            with self._lock:
                results = model.transcribe(
                    audio=[str(p) for p in paths],
                    language=lang,
                    return_time_stamps=want_ts,
                )
        except Exception as exc:  # noqa: BLE001 — surface as ASRError
            raise ASRError(f"qwen_asr transcription failed: {exc}") from exc
        if len(results) != len(paths):
            raise ASRError(
                f"qwen_asr returned {len(results)} results for {len(paths)} inputs")

        out: list[list[Utterance]] = []
        for path, res in zip(paths, results):
            text = str(getattr(res, "text", "") or "").strip()
            if not text:
                out.append([])
                continue
            words = _map_time_stamps(getattr(res, "time_stamps", None)) if want_ts else None
            out.append([Utterance(0.0, _wav_duration(path), text, words)])
        return out
