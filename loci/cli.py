"""Loci command line. Each GPU stage runs as its own subprocess so VRAM is freed between stages."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _session_dir(video: Path, session: str | None) -> Path:
    from loci.config import load_config, repo_path
    from loci.stages.extract import session_id_for

    return repo_path(load_config()["paths"]["sessions"]) / (session or session_id_for(video))


@app.command()
def extract(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    session: str | None = typer.Option(None, help="Session id (default: derived from the video)"),
    max_keyframes: int | None = typer.Option(None, "--max-keyframes", "-n"),
) -> None:
    """Stage 1: pick sharp, evenly spread keyframes and extract audio."""
    from loci.config import load_config
    from loci.stages import extract as stage

    cfg = load_config()["extract"]
    out = _session_dir(video, session)
    kfs = stage.run(video, out, max_keyframes or cfg["max_keyframes"], cfg["candidate_fps"],
                    cfg["min_sharpness_quantile"])
    typer.echo(f"{out.name}: {len(kfs)} keyframes, "
               f"{kfs[0].t:.2f}s-{kfs[-1].t:.2f}s" if kfs else f"{out.name}: no keyframes")


@app.command()
def version() -> None:
    """Print the Loci version."""
    from loci import __version__

    typer.echo(__version__)
