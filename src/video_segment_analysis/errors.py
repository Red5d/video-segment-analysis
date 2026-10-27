"""Exception hierarchy for VSA."""

from __future__ import annotations


class VSAError(Exception):
    """Base class for all VSA errors."""


class ConfigError(VSAError):
    """Invalid or missing configuration."""


class MediaError(VSAError):
    """An ffmpeg/ffprobe/yt-dlp invocation failed.

    Carries the (redacted) command and the tail of the tool's stderr.
    """

    def __init__(self, message: str, *, cmd: list[str] | None = None,
                 stderr: str | None = None) -> None:
        self.cmd = cmd
        self.stderr = stderr
        parts = [message]
        if cmd:
            parts.append("command: " + " ".join(cmd))
        if stderr:
            parts.append("stderr (tail):\n" + stderr)
        super().__init__("\n".join(parts))


class SourceError(VSAError):
    """Source resolution / probe / download failure."""


class ASRError(VSAError):
    """ASR backend failure."""


class VLMError(VSAError):
    """VLM backend failure."""


class ManifestError(VSAError):
    """manifest.json missing, corrupt, or prerequisite stage not reached."""
