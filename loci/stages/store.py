"""Stage 11: write a session into data/memory.sqlite (replacing any earlier copy of that session)."""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import numpy as np

from loci.session import Session

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
  session_id TEXT PRIMARY KEY, scene TEXT, video TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  duration_s REAL, n_keyframes INTEGER, camera_path_m REAL, up TEXT, to_room_frame TEXT, captured_at TEXT,
  start_name TEXT
);
CREATE TABLE IF NOT EXISTS keyframes (
  session_id TEXT, keyframe INTEGER, t REAL, file TEXT, area TEXT, cam2world TEXT,
  PRIMARY KEY (session_id, keyframe)
);
CREATE TABLE IF NOT EXISTS objects (
  object_id TEXT, session_id TEXT, label TEXT, aliases TEXT, area TEXT,
  cx REAL, cy REAL, cz REAL, box_min TEXT, box_max TEXT, extent_m TEXT,
  first_seen REAL, last_seen REAL, n_obs INTEGER, confidence REAL, best_obs TEXT,
  location_text TEXT, text TEXT, text_english TEXT, state TEXT, state_seen_at REAL, attribute_confidence REAL,
  distance_along_walk_m REAL, side TEXT, passed_at REAL, dist_to_door_m REAL, dist_to_start_m REAL,
  PRIMARY KEY (session_id, object_id)
);
CREATE TABLE IF NOT EXISTS observations (
  obs_id TEXT, session_id TEXT, object_id TEXT, keyframe INTEGER, t REAL, label TEXT, score REAL,
  bbox TEXT, cx REAL, cy REAL, cz REAL, box_min TEXT, box_max TEXT, dist_m REAL, quality REAL,
  PRIMARY KEY (session_id, obs_id)
);
CREATE TABLE IF NOT EXISTS relations (
  session_id TEXT, subject TEXT, relation TEXT, object TEXT, dist_m REAL
);
CREATE TABLE IF NOT EXISTS utterances (
  session_id TEXT, idx INTEGER, start_s REAL, end_s REAL, text TEXT, speaker TEXT,
  PRIMARY KEY (session_id, idx)
);
CREATE TABLE IF NOT EXISTS utterance_links (
  session_id TEXT, utterance INTEGER, object_id TEXT, dt_s REAL
);
CREATE TABLE IF NOT EXISTS alerts (
  session_id TEXT, kind TEXT, text TEXT, object_id TEXT, evidence TEXT, seen_at REAL, created_at TEXT
);
CREATE TABLE IF NOT EXISTS privacy (
  session_id TEXT, prompt TEXT, instances INTEGER, frames INTEGER
);
"""

PER_SESSION = [
    "keyframes",
    "objects",
    "observations",
    "relations",
    "utterances",
    "utterance_links",
    "privacy",
]


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    # migrations for databases created before a column existed
    cols = {r["name"] for r in con.execute("PRAGMA table_info(sessions)")}
    if "captured_at" not in cols:
        con.execute("ALTER TABLE sessions ADD COLUMN captured_at TEXT")
    if "start_name" not in cols:
        con.execute("ALTER TABLE sessions ADD COLUMN start_name TEXT")
    return con


def _j(x) -> str | None:
    return None if x is None else json.dumps(x, ensure_ascii=False)


def register_to_reference(con: sqlite3.Connection, sid: str) -> dict:
    """Align a home capture to the earliest home capture (the room frame), via objects that do not move."""
    from loci.register import register

    ref = con.execute(
        "SELECT session_id FROM sessions WHERE scene='home' ORDER BY captured_at, created_at LIMIT 1"
    ).fetchone()["session_id"]
    if ref == sid:
        res: dict = {"ok": True, "reference": sid, "transform": np.eye(4).tolist(), "n_inliers": 0}
    else:

        def objs(s: str) -> list[dict]:
            rows = con.execute("SELECT label, cx, cy, cz FROM objects WHERE session_id=?", (s,)).fetchall()
            return [{"label": r["label"], "centroid": [r["cx"], r["cy"], r["cz"]]} for r in rows]

        res = {**register(objs(sid), objs(ref)), "reference": ref}
    with con:
        con.execute(
            "UPDATE sessions SET to_room_frame=? WHERE session_id=?", (json.dumps(res) if res["ok"] else None, sid)
        )
    log.info("%s: room frame %s (%s)", sid, "ok" if res["ok"] else "not aligned", res.get("reason", res["reference"]))
    return res


def run(session: Session, db_path: Path) -> dict:
    import numpy as np

    from loci.stages.geometry import load

    meta = session.read(session.meta)
    kf_meta = session.read(session.keyframes_json)
    kfs = kf_meta["keyframes"]
    areas = session.read(session.vocab)["areas"]
    objects = session.read(session.objects)
    obs = session.read(session.observations)
    obj_of = {oid: o["object_id"] for o in objects for oid in o["obs_ids"]}
    rels_path = session.root / "relations.json"
    rels = session.read(rels_path) if rels_path.exists() else []
    utt = session.read(session.utterances) if session.utterances.exists() else {"utterances": [], "links": []}
    ledger = session.read(session.ledger)
    g = load(session)
    pose_of = {int(k): g["poses"][i] for i, k in enumerate(g["keyframe_index"])}

    con = connect(db_path)
    with con:
        sid = session.id
        con.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
        for t in PER_SESSION:
            con.execute(f"DELETE FROM {t} WHERE session_id=?", (sid,))  # noqa: S608 (fixed table names)
        con.execute(
            "INSERT INTO sessions (session_id, scene, video, duration_s, n_keyframes, camera_path_m, up, captured_at,"
            " start_name) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                sid,
                meta["scene"],
                meta.get("video"),
                kf_meta["duration_s"],
                len(kfs),
                meta.get("geometry", {}).get("camera_path_m"),
                _j(g["up"].tolist()),
                meta.get("captured_at"),
                meta.get("start_name"),
            ),
        )
        con.executemany(
            "INSERT INTO keyframes VALUES (?,?,?,?,?,?)",
            [
                (
                    sid,
                    k["index"],
                    k["t"],
                    k["file"],
                    areas[k["index"]],
                    _j(np.asarray(pose_of[k["index"]]).round(5).tolist()) if k["index"] in pose_of else None,
                )
                for k in kfs
            ],
        )
        con.executemany(
            "INSERT INTO objects VALUES (" + ",".join("?" * 27) + ")",
            [
                (
                    o["object_id"],
                    sid,
                    o["label"],
                    _j(o["aliases"]),
                    o.get("area"),
                    *o["centroid"],
                    _j(o["box_min"]),
                    _j(o["box_max"]),
                    _j(o["extent_m"]),
                    o["first_seen"],
                    o["last_seen"],
                    o["n_obs"],
                    o["confidence"],
                    o["best_obs"],
                    o.get("location_text"),
                    o.get("text"),
                    o.get("text_english"),
                    _j(o.get("state")),
                    o.get("state_seen_at"),
                    o.get("attribute_confidence"),
                    o.get("distance_along_walk_m"),
                    o.get("side"),
                    o.get("passed_at"),
                    o.get("dist_to_door_m"),
                    o.get("dist_to_start_m"),
                )
                for o in objects
            ],
        )
        con.executemany(
            "INSERT INTO observations VALUES (" + ",".join("?" * 15) + ")",
            [
                (
                    b["obs_id"],
                    sid,
                    obj_of.get(b["obs_id"]),
                    b["keyframe"],
                    b["t"],
                    b["label"],
                    b["score"],
                    _j(b["bbox"]),
                    *b["centroid"],
                    _j(b["box_min"]),
                    _j(b["box_max"]),
                    b["dist_m"],
                    b["quality"],
                )
                for b in obs
            ],
        )
        con.executemany(
            "INSERT INTO relations VALUES (?,?,?,?,?)",
            [(sid, r["subject"], r["relation"], r["object"], r["dist_m"]) for r in rels],
        )
        con.executemany(
            "INSERT INTO utterances VALUES (?,?,?,?,?,?)",
            [(sid, i, u["start_s"], u["end_s"], u["text"], u.get("speaker")) for i, u in enumerate(utt["utterances"])],
        )
        con.executemany(
            "INSERT INTO utterance_links VALUES (?,?,?,?)",
            [(sid, x["utterance"], x["object_id"], x["dt_s"]) for x in utt["links"]],
        )
        con.executemany(
            "INSERT INTO privacy VALUES (?,?,?,?)",
            [(sid, p, v["instances"], v["frames"]) for p, v in ledger["totals"].items()],
        )
    if meta["scene"] == "home":
        register_to_reference(con, sid)
    con.close()
    log.info(
        "%s: stored %d objects, %d observations, %d utterances",
        sid,
        len(objects),
        len(obs),
        len(utt["utterances"]),
    )
    return {"objects": len(objects), "observations": len(obs)}
