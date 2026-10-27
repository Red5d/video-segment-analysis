"""ASR client ABC and shared data model (DESIGN §8.2).

All timestamps are **relative to the submitted audio** (a chunk of a
segment WAV); the pipeline re-bases them to segment/absolute time.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Utterance:
    start: float
    end: float
    text: str
    words: list[Word] | None = None   # None when no aligner / API lacks timestamps


class AsrClient(ABC):
    """Transcribes one audio file into utterances with relative times."""

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> list[Utterance]:
        ...
