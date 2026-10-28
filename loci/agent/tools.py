"""Agent tools: plain functions over the SQLite memory. Objects are addressed as "<session_id>:<object_id>"."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

import numpy as np
from rapidfuzz import fuzz

SYNONYMS: dict[str, list[str]] = {
    "keys": ["key", "keychain", "car keys", "house keys", "bunch of keys"],
    "potted plant": ["plant", "plants", "houseplant", "pot plant", "flower pot"],
    "computer monitor": ["pc", "computer", "monitor", "screen", "desktop", "display"],
    "laptop": ["notebook computer", "macbook"],
    "gas stove": ["stove", "burner", "hob", "cooktop", "gas", "flame", "burner flame"],
    "mobile phone": ["phone", "cell phone", "smartphone", "mobile"],
    "wallet": ["purse"],
    "charger": ["phone charger", "charging cable", "adapter"],
    "glasses": ["specs", "spectacles", "eyeglasses"],
    "shop sign": ["shop", "store", "signboard", "sign", "board"],
    "keyboard": ["external keyboard"],
    "computer mouse": ["mouse"],
    "tv": ["television", "tv screen"],
}


def fmt_time(t: float | None) -> str | None:
    if t is None:
        return None
    m, s = divmod(int(round(t)), 60)
    return f"{m}:{s:02d}"


def uid(row: sqlite3.Row) -> str:
    return f"{row['session_id']}:{row['object_id']}"


def split_uid(u: str) -> tuple[str, str]:
    sid, _, oid = u.rpartition(":")
    return sid, oid


def expand_query(q: str) -> set[str]:
    q = q.lower().strip()
    terms = {q}
    for canon, syns in SYNONYMS.items():
        if q == canon or q in syns or any(s in q.split() for s in [canon, *syns] if " " not in s):
            terms |= {canon, *syns}
    return terms


class Memory:
    def __init__(self, con: sqlite3.Connection) -> None:
        self.con = con

    def _obj(self, u: str) -> sqlite3.Row | None:
        sid, oid = split_uid(u)
        return self.con.execute("SELECT * FROM objects WHERE session_id=? AND object_id=?", (sid, oid)).fetchone()

    def _scene(self, sid: str) -> str:
        r = self.con.execute("SELECT scene FROM sessions WHERE session_id=?", (sid,)).fetchone()
        return r["scene"] if r else "?"

    def _captured(self, sid: str) -> str | None:
        r = self.con.execute("SELECT captured_at FROM sessions WHERE session_id=?", (sid,)).fetchone()
        return r["captured_at"] if r else None

    # ---- tools -------------------------------------------------------------------------------------------

    def search_objects(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        """Fuzzy + synonym match over labels, aliases and sign text, across all sessions."""
        terms = expand_query(query)
        rows = self.con.execute("SELECT * FROM objects").fetchall()
        scored = []
        for r in rows:
            names = [r["label"], *json.loads(r["aliases"] or "[]")]
            texts = [t for t in (r["text"], r["text_english"]) if t]
            s_name = max(fuzz.WRatio(t, n.lower()) for t in terms for n in names)
            s_text = max((fuzz.WRatio(query.lower(), t.lower()) for t in texts), default=0)
            score = max(s_name, s_text)
            if score >= 80:
                scored.append((score, r))
        scored.sort(key=lambda x: (-x[0], -x[1]["last_seen"]))
        return [
            {
                "id": uid(r),
                "label": r["label"],
                "text": r["text"],
                "session": r["session_id"],
                "scene": self._scene(r["session_id"]),
                "area": r["area"],
                "location_text": r["location_text"],
                "last_seen": fmt_time(r["last_seen"]),
                "confidence": r["confidence"],
                "match": round(s, 1),
            }
            for s, r in scored[:limit]
        ]

    def get_object(self, id: str) -> dict[str, Any]:
        r = self._obj(id)
        if r is None:
            return {"error": f"no object {id}"}
        sid = r["session_id"]
        rels = self.con.execute(
            "SELECT relation, object, dist_m FROM relations WHERE session_id=? AND subject=?",
            (sid, r["object_id"]),
        ).fetchall()
        evidence = [
            o["obs_id"]
            for o in self.con.execute(
                "SELECT obs_id FROM observations WHERE session_id=? AND object_id=? ORDER BY quality DESC LIMIT 3",
                (sid, r["object_id"]),
            )
        ]
        out = {
            "id": id,
            "label": r["label"],
            "scene": self._scene(sid),
            "area": r["area"],
            "position_m": [round(r["cx"], 3), round(r["cy"], 3), round(r["cz"], 3)],
            "extent_m": json.loads(r["extent_m"]),
            "location_text": r["location_text"],
            "relations": [
                {"relation": x["relation"], "object": f"{sid}:{x['object']}", "dist_m": x["dist_m"]} for x in rels
            ],
            "first_seen": fmt_time(r["first_seen"]),
            "last_seen": fmt_time(r["last_seen"]),
            "n_sightings": r["n_obs"],
            "confidence": r["confidence"],
            "evidence": evidence,
        }
        for k in (
            "text",
            "text_english",
            "side",
            "distance_along_walk_m",
            "dist_to_door_m",
            "dist_to_start_m",
        ):
            if r[k] is not None:
                out[k] = r[k]
        if r["passed_at"] is not None:
            out["passed_at"] = fmt_time(r["passed_at"])
        return out

    def get_state(self, id: str) -> dict[str, Any]:
        r = self._obj(id)
        if r is None:
            return {"error": f"no object {id}"}
        if r["state"] is None:
            return {"id": id, "state": None, "note": "no state recorded for this object"}
        obs = self.con.execute(
            "SELECT obs_id FROM observations WHERE session_id=? AND object_id=? ORDER BY ABS(t - ?) LIMIT 1",
            (r["session_id"], r["object_id"], r["state_seen_at"]),
        ).fetchone()
        return {
            "id": id,
            "state": json.loads(r["state"]),
            "seen_at": fmt_time(r["state_seen_at"]),
            "confidence": r["attribute_confidence"],
            "evidence": [obs["obs_id"]] if obs else [],
            "note": "state as of seen_at; not known now",
        }

    def state_history(self, query: str) -> list[dict[str, Any]]:
        """Recorded states of matching objects in every capture, newest capture first."""
        out = []
        for h in self.search_objects(query, limit=20):
            st = self.get_state(h["id"])
            if st.get("state") is None:
                continue
            sid = h["session"]
            out.append(
                {
                    "id": h["id"],
                    "label": h["label"],
                    "capture": sid,
                    "captured_at": self._captured(sid),
                    "state": st["state"],
                    "seen_at": st["seen_at"],
                    "confidence": st["confidence"],
                    "evidence": st["evidence"],
                }
            )
        return sorted(out, key=lambda x: x["captured_at"] or "", reverse=True)

    def route(self, from_id: str, to_id: str) -> dict[str, Any]:
        """Walking directions between two things seen on the same walk, following the path you took."""
        a, b = self._obj(from_id), self._obj(to_id)
        if a is None or b is None:
            return {"error": f"unknown object {from_id if a is None else to_id}"}
        if a["session_id"] != b["session_id"]:
            return {"error": "the two places were seen in different captures"}
        sid = a["session_id"]
        da, db = a["distance_along_walk_m"], b["distance_along_walk_m"]
        if da is None or db is None:  # home: straight line only
            d = float(np.linalg.norm(np.array([a["cx"], a["cy"], a["cz"]]) - np.array([b["cx"], b["cy"], b["cz"]])))
            return {"from": from_id, "to": to_id, "straight_line_m": round(d, 1)}
        lo, hi = sorted((da, db))
        between = self.con.execute(
            "SELECT * FROM objects WHERE session_id=? AND distance_along_walk_m > ? AND distance_along_walk_m < ? "
            "AND (text IS NOT NULL OR label LIKE '%stall%' OR label LIKE '%shop%') ORDER BY distance_along_walk_m",
            (sid, lo, hi),
        ).fetchall()
        if db < da:
            between = list(reversed(between))
        return {
            "from": from_id,
            "to": to_id,
            "walk_m": round(abs(db - da), 1),
            "direction": "same way you walked" if db >= da else "back the way you came",
            "destination_side": f"on your {b['side']}"
            if db >= da
            else f"on your {'left' if b['side'] == 'right' else 'right'}",
            "you_pass": [
                {"name": r["text"] or r["label"], "side": r["side"], "at_m": r["distance_along_walk_m"]}
                for r in between
            ],
            "evidence": [b["best_obs"]],
        }

    def places_passed(self, session_id: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT * FROM objects WHERE text IS NOT NULL AND distance_along_walk_m IS NOT NULL"
        args: tuple = ()
        if session_id:
            q += " AND session_id=?"
            args = (session_id,)
        rows = self.con.execute(q + " ORDER BY session_id, distance_along_walk_m", args).fetchall()
        return [
            {
                "id": uid(r),
                "name": r["text"],
                "name_english": r["text_english"],
                "distance_along_walk_m": r["distance_along_walk_m"],
                "side": r["side"],
                "passed_at": fmt_time(r["passed_at"]),
                "evidence": [r["best_obs"]],
            }
            for r in rows
        ]

    def objects_near(self, id: str, radius_m: float = 1.0) -> list[dict[str, Any]]:
        r = self._obj(id)
        if r is None:
            return [{"error": f"no object {id}"}]
        c = np.array([r["cx"], r["cy"], r["cz"]])
        out = []
        for o in self.con.execute("SELECT * FROM objects WHERE session_id=?", (r["session_id"],)):
            if o["object_id"] == r["object_id"]:
                continue
            d = float(np.linalg.norm(np.array([o["cx"], o["cy"], o["cz"]]) - c))
            if d <= radius_m:
                out.append({"id": uid(o), "label": o["label"], "dist_m": round(d, 2)})
        return sorted(out, key=lambda x: x["dist_m"])

    def objects_on(self, anchor_id: str) -> list[dict[str, Any]]:
        sid, aid = split_uid(anchor_id)
        rows = self.con.execute(
            "SELECT o.* FROM relations r JOIN objects o ON o.session_id=r.session_id AND o.object_id=r.subject "
            "WHERE r.session_id=? AND r.object=? AND r.relation IN ('on','inside')",
            (sid, aid),
        ).fetchall()
        return [
            {
                "id": uid(o),
                "label": o["label"],
                "position_m": [round(o["cx"], 3), round(o["cy"], 3), round(o["cz"], 3)],
                "location_text": o["location_text"],
                "evidence": [o["best_obs"]],
            }
            for o in rows
        ]

    def history(self, label: str) -> list[dict[str, Any]]:
        """Every sighting of objects matching label, across sessions, newest session first."""
        hits = self.search_objects(label, limit=20)
        return sorted(hits, key=lambda h: h["session"], reverse=True)

    def search_notes(self, text: str) -> list[dict[str, Any]]:
        out = []
        for u in self.con.execute("SELECT * FROM utterances").fetchall():
            score = fuzz.partial_ratio(text.lower(), u["text"].lower())
            links = self.con.execute(
                "SELECT object_id, dt_s FROM utterance_links WHERE session_id=? AND utterance=? ORDER BY dt_s",
                (u["session_id"], u["idx"]),
            ).fetchall()
            linked_labels = [
                self._obj(f"{u['session_id']}:{x['object_id']}")["label"]
                for x in links  # type: ignore[index]
            ]
            label_hit = max((fuzz.partial_ratio(text.lower(), lab) for lab in linked_labels), default=0)
            if max(score, label_hit) >= 60:
                out.append(
                    {
                        "session": u["session_id"],
                        "said_at": fmt_time(u["start_s"]),
                        "text": u["text"],
                        "objects_in_view": [f"{u['session_id']}:{x['object_id']}" for x in links[:5]],
                        "match": max(score, label_hit),
                    }
                )
        return sorted(out, key=lambda x: -x["match"])

    def get_evidence(self, observation_id: str, session_id: str) -> dict[str, Any]:
        o = self.con.execute(
            "SELECT * FROM observations WHERE session_id=? AND obs_id=?", (session_id, observation_id)
        ).fetchone()
        if o is None:
            return {"error": f"no observation {observation_id} in {session_id}"}
        return {
            "obs_id": o["obs_id"],
            "object": f"{session_id}:{o['object_id']}",
            "label": o["label"],
            "seen_at": fmt_time(o["t"]),
            "score": o["score"],
            "image": f"/evidence/{session_id}/{o['obs_id']}.jpg",
        }
