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
    s.write(
        s.meta,
        {
            "session_id": s.id,
            "scene": scene.value,
            "video": str(video.resolve()),
            "captured_at": stage.captured_at(video),
        },
    )
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
def track(session: str, debug_video: bool = typer.Option(True, help="write debug/<session>_tracks.mp4")) -> None:
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
def geometry(session: str) -> None:
    """Stage 5: MapAnything poses, metric depth and scene.ply."""
    from loci.config import repo_path
    from loci.stages import geometry as stage

    s = _session(session)
    res = stage.run(s, str(repo_path(_cfg()["models"]["mapanything"]["checkpoint"])))
    typer.echo(f"{s.id}: {res['n_views']} views, path {res['camera_path_m']} m, {res['ply_points']} points")


@app.command()
def lift(session: str) -> None:
    """Stage 6: lift detections to 3D (CPU)."""
    from loci.stages import lift as stage

    s = _session(session)
    typer.echo(f"{s.id}: {len(stage.run(s))} observations")


@app.command()
def associate(session: str) -> None:
    """Stage 7: merge observations into objects (CPU)."""
    from loci.stages import associate as stage

    s = _session(session)
    typer.echo(f"{s.id}: {len(stage.run(s))} objects")


@app.command()
def relations(session: str) -> None:
    """Stage 8: spatial relations and location text (CPU)."""
    from loci.stages import relations as stage

    s = _session(session)
    res = stage.run(s)
    typer.echo(f"{s.id}: {len(res['relations'])} relations")


@app.command()
def attributes(session: str) -> None:
    """Stage 9: Glimmer reads signs and checks object states."""
    from loci.llm import glimmer_server
    from loci.providers import get_vision_checker
    from loci.stages import attributes as stage

    s = _session(session)
    if stage.pending(s) == 0:
        res = stage.run(s, None)  # everything cached: no Glimmer start
    else:
        with glimmer_server(s.root / "glimmer.log") as url:
            res = stage.run(s, get_vision_checker(url))
    typer.echo(f"{s.id}: {len(res)} attributes")


@app.command()
def transcribe(session: str) -> None:
    """Stage 10: transcribe audio and link utterances to objects in view."""
    from loci.providers import get_transcriber
    from loci.stages import transcribe as stage

    s = _session(session)
    res = stage.run(s, get_transcriber())
    for u in res["utterances"]:
        typer.echo(f"  {u['start_s']:6.1f}s  {u['text']}")


@app.command()
def store(session: str) -> None:
    """Stage 11: write the session into data/memory.sqlite and print its object table."""
    from loci.config import repo_path
    from loci.stages import store as stage

    s = _session(session)
    stage.run(s, repo_path(_cfg()["paths"]["memory_db"]))
    _print_objects(s.id)


def _print_objects(session_id: str) -> None:
    from loci.config import repo_path
    from loci.stages.store import connect

    con = connect(repo_path(_cfg()["paths"]["memory_db"]))
    rows = con.execute(
        "SELECT object_id, label, n_obs, confidence, cx, cy, cz, location_text, text, state, state_seen_at "
        "FROM objects WHERE session_id=? ORDER BY first_seen",
        (session_id,),
    ).fetchall()
    typer.echo(f"\n{session_id}: {len(rows)} objects")
    typer.echo(f"{'object':24} {'n':>3} {'conf':>5}  {'position (m)':22} location / text / state")
    for r in rows:
        extra = r["text"] or (f"{r['state']} @ {r['state_seen_at']:.1f}s" if r["state"] else "")
        pos = f"[{r['cx']:6.2f} {r['cy']:6.2f} {r['cz']:6.2f}]"
        typer.echo(
            f"{r['object_id']:24} {r['n_obs']:3d} {r['confidence']:5.2f}  {pos:22} {r['location_text'] or ''} {extra}"
        )
    con.close()


# stage name -> output file (relative to the session folder) that marks it done
STAGES = [
    ("redact", "privacy_ledger.json"),
    ("name", "vocab.json"),
    ("track", "tracks.json"),
    ("geometry", "geometry.npz"),
    ("lift", "observations.json"),
    ("associate", "objects.json"),
    ("relations", "relations.json"),
    ("attributes", "attributes.json"),
    ("transcribe", "utterances.json"),
    ("store", None),
    ("check", None),
    ("coach", None),
]


@app.command()
def build(
    video: Path = typer.Argument(..., exists=True, dir_okay=False),
    scene: Scene = typer.Option(Scene.home, "--scene"),
    session: str | None = typer.Option(None, help="Session id (default: derived from the video)"),
    from_stage: str | None = typer.Option(None, "--from", help="re-run this stage and everything after it"),
) -> None:
    """Run every stage on a video, each in its own subprocess (VRAM is freed between stages)."""
    import subprocess
    import sys
    import time

    from loci.session import Session
    from loci.stages.extract import session_id_for

    sid = session or session_id_for(video)
    s = Session.named(sid)
    names = [n for n, _ in STAGES]
    if from_stage and from_stage not in ["extract", *names]:
        raise typer.BadParameter(f"--from must be one of extract, {', '.join(names)}")
    rerun = False if from_stage else None

    def run(args: list[str]) -> None:
        t = time.perf_counter()
        res = subprocess.run([sys.executable, "-m", "loci", *args])
        if res.returncode != 0:
            raise typer.Exit(res.returncode)
        log.info("stage %s done in %.0fs", args[0], time.perf_counter() - t)

    if from_stage == "extract" or not (s.root / "keyframes").exists() and not s.raw_dir.exists():
        run(["extract", str(video), "--scene", scene.value, "--session", sid])
    for stage, marker in STAGES:
        if from_stage == stage:
            rerun = True
        done = marker is not None and (s.root / marker).exists()
        if done and not rerun:
            log.info("stage %s: cached", stage)
            continue
        run([stage, sid])


@app.command()
def ask(
    question: str,
    no_cloud: bool = typer.Option(True, "--no-cloud/--cloud", help="Phase 1: local only"),
    show_trace: bool = typer.Option(False, "--trace"),
) -> None:
    """Ask the memory a question (searches every session)."""
    import json

    from loci.agent.agent import ask as run_ask
    from loci.agent.tools import Memory
    from loci.config import repo_path
    from loci.llm import glimmer_server
    from loci.stages.store import connect

    con = connect(repo_path(_cfg()["paths"]["memory_db"]))
    with glimmer_server(repo_path("debug") / "glimmer_ask.log") as url:
        ans = run_ask(question, Memory(con), url)
    trace = ans.pop("trace", [])
    typer.echo(f"\n{ans['answer']}\n")
    typer.echo(json.dumps({k: v for k, v in ans.items() if k != "answer"}, indent=2, ensure_ascii=False))
    if show_trace:
        typer.echo(json.dumps(trace, indent=2, ensure_ascii=False)[:8000])


@app.command()
def check(session: str) -> None:
    """Leaving check: the agent flags what is worth a heads-up (states, valuables, spoken notes)."""
    from loci.agent.leaving import run as leaving
    from loci.agent.tools import Memory
    from loci.config import repo_path
    from loci.llm import glimmer_server
    from loci.stages.store import connect

    mem = Memory(connect(repo_path(_cfg()["paths"]["memory_db"])))
    with glimmer_server(repo_path("debug") / "glimmer_check.log") as url:
        alerts = leaving(mem, session, url)
    typer.echo(f"{session}: {len(alerts)} alerts")
    for a in alerts:
        typer.echo(f"  [{a.get('kind')}] {a['text']}  ({', '.join(a.get('evidence', []))})")


@app.command(name="coach")
def coach_cmd(session: str) -> None:
    """Capture quality: is the 3D trustworthy, and what to re-walk next time (no models needed)."""
    from loci.quality_run import run as quality_run

    s = _session(session)
    q = quality_run(s)
    r = q["reconstruction"]
    typer.echo(f"{s.id}: 3D {r['verdict']} (bad steps {r['bad_step_fraction']:.0%}, max {r['max_speed_mps']} m/s)")
    for tip in q["tips"]:
        typer.echo(f"  - {tip}")


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """API + viewer at http://host:port (starts Glimmer, or reuses a running one)."""
    import uvicorn

    uvicorn.run("loci.api:app", host=host, port=port, log_level="info")


@app.command()
def objects(session: str) -> None:
    """Print a session's object table from the memory."""
    _print_objects(session)


@app.command()
def version() -> None:
    """Print the Loci version."""
    from loci import __version__

    typer.echo(__version__)
