"""Loci command line. Each GPU stage runs as its own subprocess so VRAM is freed between stages."""

from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path

import typer

app = typer.Typer(no_args_is_help=True, add_completion=False)
log = logging.getLogger("loci")


class Scene(StrEnum):
    home = "home"
    street = "street"


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v")) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _cfg() -> dict:
    from loci.config import load_config

    return load_config()


def _session(session_id: str):
    from loci.session import Session

    s = Session.named(session_id)
    if not s.root.exists():
        raise typer.BadParameter(f"no session folder {s.root}")
    return s


@app.command()
def extract(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    scene: Scene = typer.Option(Scene.home, "--scene"),
    session: str | None = typer.Option(None, help="Session id (default: derived from the video)"),
    max_keyframes: int | None = typer.Option(None, "--max-keyframes", "-n"),
) -> None:
    """Stage 1: pick sharp, evenly spread keyframes and extract audio."""
    from loci.session import Session
    from loci.stages import extract as stage

    cfg = _cfg()["extract"]
    s = Session.named(session or stage.session_id_for(video))
    kfs = stage.run(
        video,
        s.root,
        max_keyframes or cfg["max_keyframes"],
        cfg["candidate_fps"],
        cfg["min_sharpness_quantile"],
    )
    link = s.root / "video.mp4"
    if not link.exists():
        link.symlink_to(video.resolve())
    s.write(s.meta, {"session_id": s.id, "scene": scene.value, "video": str(video.resolve())})
    typer.echo(f"{s.id}: {len(kfs)} keyframes, {kfs[0].t:.2f}s-{kfs[-1].t:.2f}s" if kfs else f"{s.id}: none")


@app.command()
def redact(session: str, keep_raw: bool = typer.Option(False, "--keep-raw")) -> None:
    """Stage 2: blur faces, plates, screens, documents (SAM 3.1); delete raw frames."""
    from loci.config import repo_path
    from loci.stages import redact as stage

    s = _session(session)
    cfg = _cfg()
    prompts = cfg["scenes"][s.scene()]["redact_prompts"]
    res = stage.run(s, prompts, str(repo_path(cfg["models"]["sam3"]["checkpoint"])), keep_raw=keep_raw)
    typer.echo(f"{s.id}: " + ", ".join(f"{p} {v['instances']}" for p, v in res["totals"].items()))


@app.command()
def name(session: str) -> None:
    """Stage 3: Glimmer names objects and labels areas (redacted frames only)."""
    from loci.llm import glimmer_server
    from loci.providers import get_namer
    from loci.stages import name as stage

    s = _session(session)
    vocab = _cfg()["scenes"][s.scene()]["vocab"]
    with glimmer_server(s.root / "glimmer.log") as url:
        res = stage.run(s, get_namer(url), vocab)
    typer.echo(f"{s.id}: {len(res['vocab'])} phrases; areas {sorted(set(res['areas']))}")
    typer.echo(", ".join(res["vocab"]))


@app.command()
def track(
    session: str, debug_video: bool = typer.Option(True, help="write debug/<session>_tracks.mp4")
) -> None:
    """Stage 4: SAM 3.1 finds every vocabulary phrase in every keyframe."""
    from loci.config import repo_path
    from loci.stages import track as stage

    s = _session(session)
    cfg = _cfg()
    phrases = s.read(s.vocab)["vocab"]
    exclude = cfg["scenes"][s.scene()].get("exclude_from_pointcloud", [])
    res = stage.run(
        s,
        phrases,
        exclude,
        str(repo_path(cfg["models"]["sam3"]["checkpoint"])),
        min_score=cfg["track"]["min_score"],
    )
    if debug_video:
        out = repo_path("debug") / f"{s.id}_tracks.mp4"
        out.parent.mkdir(exist_ok=True)
        stage.overlay_video(s, out)
        typer.echo(f"overlay: {out}")
    counts: dict[str, int] = {}
    for inst in res["instances"]:
        counts[inst["label"]] = counts.get(inst["label"], 0) + 1
    typer.echo(f"{s.id}: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda x: -x[1])))


@app.command()
def version() -> None:
    """Print the Loci version."""
    from loci import __version__

    typer.echo(__version__)
