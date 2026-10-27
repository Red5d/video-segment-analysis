"""Typer CLI: `vsa segment | transcribe | slides | all` (DESIGN §4)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import Config, load_config
from .errors import VSAError
from .log import setup_logging

app = typer.Typer(
    name="vsa",
    help="Split long conference recordings into per-talk segments, "
         "transcribe them, and describe slide changes.",
    no_args_is_help=True,
)
logger = logging.getLogger("vsa.cli")


def _setup(config: Optional[Path], out: Optional[Path]) -> Config:
    """Load config (+ --out override), init logging, verify ffmpeg."""
    overrides: dict = {}
    if out is not None:
        overrides["paths"] = {"workdir": str(out)}
    cfg = load_config(config, overrides=overrides)
    log_file = None
    if cfg.logging.file:
        log_file = cfg.paths.workdir / cfg.logging.file
    setup_logging(cfg.logging.level, log_file)
    from .ffmpegio import check_ffmpeg
    check_ffmpeg()
    return cfg


def _die(exc: VSAError) -> None:
    logger.error("%s", exc)
    raise typer.Exit(code=1)


@app.command()
def segment(
    source: str = typer.Option(..., "--source",
                               help="Local video path or http(s) URL (yt-dlp)."),
    config: Optional[Path] = typer.Option(None, "--config", help="config.yaml path."),
    out: Optional[Path] = typer.Option(None, "--out", help="Output directory."),
    strategy: Optional[str] = typer.Option(
        None, "--strategy", help="auto|local-copy|local-encode|yt-sections"),
    segment_id: Optional[list[str]] = typer.Option(
        None, "--segment-id", help="Restrict cutting to these segment ids."),
    force: bool = typer.Option(False, "--force", help="Redo stages regardless."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan; exit 0."),
) -> None:
    """Process 1: obtain audio, detect speech, group into segments, cut media."""
    cfg = _setup(config, out)
    from .segment_pipeline import run_segment
    try:
        code = run_segment(cfg, source, strategy=strategy,
                           segment_ids=list(segment_id or []),
                           force=force, dry_run=dry_run)
    except VSAError as exc:
        _die(exc)
    raise typer.Exit(code=code)


@app.command()
def transcribe(
    config: Optional[Path] = typer.Option(None, "--config"),
    out: Optional[Path] = typer.Option(None, "--out"),
    backend: Optional[str] = typer.Option(
        None, "--backend", help="qwen_local|openai_compat|fake (default from config)."),
    segment_id: Optional[list[str]] = typer.Option(
        None, "--segment-id", help="Transcribe only these segments (repair runs)."),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Process 2: transcribe cut segments with Qwen3-ASR (or a stub) → SRT/VTT."""
    cfg = _setup(config, out)
    if backend:
        cfg.asr.backend = backend  # type: ignore[assignment]
    from .transcribe_pipeline import run_transcribe
    try:
        code = run_transcribe(cfg, segment_ids=list(segment_id or []),
                              force=force, dry_run=dry_run)
    except VSAError as exc:
        _die(exc)
    raise typer.Exit(code=code)


@app.command()
def slides(
    config: Optional[Path] = typer.Option(None, "--config"),
    out: Optional[Path] = typer.Option(None, "--out"),
    change_threshold: Optional[float] = typer.Option(
        None, "--change-threshold", help="slides.change_ratio override."),
    min_change_interval: Optional[float] = typer.Option(
        None, "--min-change-interval", help="slides.min_change_interval override (s)."),
    segment_id: Optional[list[str]] = typer.Option(
        None, "--segment-id", help="Process only these segments (repair runs)."),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Process 3: detect slide changes, extract frames, describe with a VLM → SRT/VTT."""
    cfg = _setup(config, out)
    if change_threshold is not None:
        cfg.slides.change_ratio = change_threshold
    if min_change_interval is not None:
        cfg.slides.min_change_interval = min_change_interval
    from .slides_pipeline import run_slides
    try:
        code = run_slides(cfg, segment_ids=list(segment_id or []),
                          force=force, dry_run=dry_run)
    except VSAError as exc:
        _die(exc)
    raise typer.Exit(code=code)


@app.command("all")
def all_(
    source: str = typer.Option(..., "--source"),
    config: Optional[Path] = typer.Option(None, "--config"),
    out: Optional[Path] = typer.Option(None, "--out"),
    strategy: Optional[str] = typer.Option(None, "--strategy"),
    backend: Optional[str] = typer.Option(None, "--backend"),
    segment_id: Optional[list[str]] = typer.Option(None, "--segment-id"),
    force: bool = typer.Option(False, "--force"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Run segment → transcribe → slides in order."""
    cfg = _setup(config, out)
    if backend:
        cfg.asr.backend = backend  # type: ignore[assignment]
    from .segment_pipeline import run_segment
    from .slides_pipeline import run_slides
    from .transcribe_pipeline import run_transcribe
    try:
        codes = [run_segment(cfg, source, strategy=strategy,
                             segment_ids=list(segment_id or []),
                             force=force, dry_run=dry_run)]
        if not dry_run:
            codes.append(run_transcribe(cfg, segment_ids=list(segment_id or []),
                                        force=force, dry_run=dry_run))
            codes.append(run_slides(cfg, segment_ids=list(segment_id or []),
                                    force=force, dry_run=dry_run))
    except VSAError as exc:
        _die(exc)
    raise typer.Exit(code=1 if 1 in codes else (2 if 2 in codes else 0))


@app.command()
def version() -> None:
    """Print the VSA version."""
    typer.echo(__version__)


def main() -> None:  # console entry point
    app()


if __name__ == "__main__":
    main()
