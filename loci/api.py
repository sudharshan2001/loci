"""FastAPI server for the viewer. Run with `loci serve`.

Glimmer is started on the first question and kept for the lifetime of the server (or an already-running
one is reused); browsing scenes, evidence and the privacy ledger needs no GPU. Spoken
questions are transcribed on the CPU so they never compete with Glimmer for GPU memory.
"""

from __future__ import annotations

import json
import logging
import tempfile
import threading
from contextlib import ExitStack, asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from loci.agent.agent import ask
from loci.agent.tools import Memory, fmt_time
from loci.config import load_config, repo_path
from loci.evidence import render
from loci.session import Session
from loci.stages.store import connect

log = logging.getLogger(__name__)
STATE: dict[str, Any] = {}


def _db() -> Memory:
    return Memory(connect(repo_path(load_config()["paths"]["memory_db"])))


_GLIMMER_LOCK = threading.Lock()
_STACK = ExitStack()


def _glimmer() -> str:
    """Start Glimmer on the first question (or reuse a running server), so browsing needs no GPU."""
    from loci.llm import glimmer_server

    with _GLIMMER_LOCK:
        if "glimmer" not in STATE:
            STATE["glimmer"] = _STACK.enter_context(glimmer_server(repo_path("debug") / "glimmer_api.log"))
    return STATE["glimmer"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        yield
    finally:
        _STACK.close()  # stops Glimmer if this server started it


app = FastAPI(title="Loci", lifespan=lifespan)


class Question(BaseModel):
    question: str


def _session(sid: str) -> Session:
    s = Session.named(sid)
    if not s.root.exists():
        raise HTTPException(404, f"no session {sid}")
    return s


@app.post("/ask")
def ask_text(q: Question) -> dict[str, Any]:
    ans = ask(q.question, _db(), _glimmer())
    ans["question"] = q.question
    return ans


@app.post("/ask_audio")
async def ask_audio(audio: UploadFile = File(...)) -> dict[str, Any]:
    import subprocess

    from loci.providers.whisper import FasterWhisperTranscriber

    with tempfile.TemporaryDirectory() as d:
        raw, wav = Path(d) / "in", Path(d) / "q.wav"
        raw.write_bytes(await audio.read())
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", str(raw), "-ac", "1", "-ar", "16000", str(wav)], check=True
        )
        if "transcriber" not in STATE:
            STATE["transcriber"] = FasterWhisperTranscriber(load_config()["models"]["whisper"]["size"], device="cpu")
        utts = STATE["transcriber"].transcribe(wav)
    question = " ".join(u.text for u in utts).strip()
    if not question:
        raise HTTPException(422, "no speech recognised")
    ans = ask(question, _db(), _glimmer())
    ans["question"] = question
    ans["transcriber"] = "faster-whisper (local)"
    return ans


@app.get("/sessions")
def sessions() -> list[dict[str, Any]]:
    mem = _db()
    rows = mem.con.execute(
        "SELECT s.*, (SELECT COUNT(*) FROM objects o WHERE o.session_id=s.session_id) AS n_objects "
        "FROM sessions s ORDER BY created_at"
    ).fetchall()
    return [
        {
            "session_id": r["session_id"],
            "scene": r["scene"],
            "n_objects": r["n_objects"],
            "duration_s": r["duration_s"],
            "camera_path_m": r["camera_path_m"],
        }
        for r in rows
    ]


@app.get("/scene/{sid}.ply")
def scene_ply(sid: str) -> FileResponse:
    return FileResponse(_session(sid).ply, media_type="application/octet-stream")


@app.get("/scene/{sid}")
def scene(sid: str) -> dict[str, Any]:
    """Camera path, 'you are here' pose, object boxes and (street) shop markers for the viewer."""
    s = _session(sid)
    g = np.load(s.geometry)
    poses = g["poses"]
    objs = s.read(s.objects)
    kfs = s.keyframes()
    return {
        "session_id": sid,
        "scene": s.scene(),
        "camera_path": poses[:, :3, 3].round(3).tolist(),
        "camera_times": [k["t"] for k in kfs],
        "you_are_here": poses[-1].round(4).tolist(),
        "up": g["up"].tolist(),
        "objects": [
            {
                k: o.get(k)
                for k in (
                    "object_id",
                    "label",
                    "centroid",
                    "box_min",
                    "box_max",
                    "location_text",
                    "text",
                    "side",
                    "distance_along_walk_m",
                    "state",
                    "best_obs",
                )
            }
            for o in objs
        ],
    }


@app.get("/objects")
def objects(session: str | None = None) -> list[dict[str, Any]]:
    mem = _db()
    q, args = "SELECT * FROM objects", ()
    if session:
        q, args = q + " WHERE session_id=?", (session,)
    return [
        {
            "id": f"{r['session_id']}:{r['object_id']}",
            "label": r["label"],
            "area": r["area"],
            "location_text": r["location_text"],
            "text": r["text"],
            "last_seen": fmt_time(r["last_seen"]),
            "position_m": [r["cx"], r["cy"], r["cz"]],
            "confidence": r["confidence"],
        }
        for r in mem.con.execute(q, args).fetchall()
    ]


@app.get("/evidence/{sid}/{obs_id}.jpg")
def evidence(sid: str, obs_id: str, full: bool = False) -> FileResponse:
    try:
        return FileResponse(render(_session(sid), obs_id, full=full), media_type="image/jpeg")
    except KeyError as e:
        raise HTTPException(404, f"no observation {obs_id}") from e


@app.get("/privacy/{sid}")
def privacy(sid: str) -> dict[str, Any]:
    s = _session(sid)
    led = s.read(s.ledger)
    return {
        "session_id": sid,
        "n_keyframes": led["n_keyframes"],
        "blurred": {p: v["instances"] for p, v in led["totals"].items()},
        "frames_with_blur": sum(1 for f in led["frames"] if f["blurred_fraction"] > 0),
    }


@app.get("/alerts/{sid}")
def alerts(sid: str) -> list[dict[str, Any]]:
    mem = _db()
    rows = mem.con.execute("SELECT * FROM alerts WHERE session_id=? ORDER BY seen_at", (sid,)).fetchall()
    return [{**dict(r), "evidence": json.loads(r["evidence"] or "[]")} for r in rows]


VIEWER = repo_path("viewer")
if VIEWER.exists():
    app.mount("/", StaticFiles(directory=VIEWER, html=True), name="viewer")
