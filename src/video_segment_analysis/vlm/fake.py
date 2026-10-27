"""Fake VLM backend: deterministic fixture text (tests, ``--dry-run``)."""

from __future__ import annotations

from pathlib import Path

from .base import VlmClient


class FakeVlmClient(VlmClient):
    """Returns ``"Slide content fixture #<n> for <stem>"``. The number is
    assigned per image path on first call and stable across calls, so
    re-describing a frame is idempotent."""

    name = "fake"

    def __init__(self) -> None:
        self._seen: dict[Path, int] = {}

    def describe(self, image_path: Path) -> str:
        p = Path(image_path)
        if p not in self._seen:
            self._seen[p] = len(self._seen) + 1
        return f"Slide content fixture #{self._seen[p]} for {p.stem}"
