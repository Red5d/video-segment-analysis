"""VLM client ABC (DESIGN §9.4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

#: Placeholder cue text when a frame could not be described.
UNAVAILABLE_TEXT = "[slide content unavailable]"


@dataclass
class SlideDescription:
    text: str
    failed: bool = False


class VlmClient(ABC):
    """Vision-LLM backend: describe the slide shown in a JPEG frame."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Backend identifier (for logs/manifest)."""

    @abstractmethod
    def describe(self, image_path: Path) -> str:
        """Cleaned plain-text description of the slide in ``image_path``."""

    def describe_checked(self, image_path: Path) -> SlideDescription:
        """``describe()`` plus a failure flag (backends override where the
        distinction is cheap; a failure must yield ``UNAVAILABLE_TEXT``)."""
        return SlideDescription(self.describe(image_path))
