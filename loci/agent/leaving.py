"""Leaving check: after a capture, the agent decides on its own what is worth flagging.

Code gathers the facts (recorded states, small personal items, spoken notes); Glimmer chooses which deserve
an alert and writes each one with evidence; every alert goes through the same verifier as answers. State
alerts always say when the state was seen ("as of 0:31"), never what it is now.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from loci.agent.tools import Memory, fmt_time
from loci.agent.verify import verify
from loci.llm import chat_json

log = logging.getLogger(__name__)

PERSONAL = {
    "keys",
    "wallet",
    "charger",
    "phone",
    "mobile phone",
    "passport",
    "glasses",
    "watch",
    "earphones",
    "headphones",
    "umbrella",
    "bag",
    "backpack",
    "id card",
    "purse",
    "laptop",
}

PROMPT = """You are Loci's leaving check. The user just finished walking through a place (capture {sid}). \
Below are facts from that capture. Decide which ones are worth a short heads-up as they leave, for example a \
burner or iron left on, a tap running, a door or window left open, small valuables left somewhere, or a \
promise they said out loud ("water this tomorrow"). Skip anything ordinary. At most 4 alerts.

Rules: state alerts must say when it was seen, e.g. "The left burner was lit when you left the kitchen at \
0:31". Never claim the state now. Use only ids and observation ids from the facts.

Facts:
{facts}

Reply with only JSON: {{"alerts": [{{"kind": "state|item|note", "text": str, "object_id": str, \
"evidence": [observation ids]}}]}}"""


def facts(mem: Memory, sid: str) -> list[dict[str, Any]]:
    out = []
    for r in mem.con.execute("SELECT * FROM objects WHERE session_id=?", (sid,)).fetchall():
        uid = f"{sid}:{r['object_id']}"
        if r["state"]:
            st = mem.get_state(uid)
            out.append(
                {
                    "kind": "state",
                    "id": uid,
                    "label": r["label"],
                    "area": r["area"],
                    "state": st["state"],
                    "seen_at": st["seen_at"],
                    "confidence": st["confidence"],
                    "evidence": st["evidence"],
                }
            )
        elif any(p == r["label"] or p in r["label"].split() for p in PERSONAL):
            out.append(
                {
                    "kind": "item",
                    "id": uid,
                    "label": r["label"],
                    "where": r["location_text"],
                    "last_seen": fmt_time(r["last_seen"]),
                    "evidence": [r["best_obs"]],
                }
            )
    for u in mem.con.execute("SELECT * FROM utterances WHERE session_id=?", (sid,)).fetchall():
        links = mem.con.execute(
            "SELECT object_id FROM utterance_links WHERE session_id=? AND utterance=? ORDER BY dt_s LIMIT 3",
            (sid, u["idx"]),
        ).fetchall()
        if links:
            first = f"{sid}:{links[0]['object_id']}"
            ev = mem.get_object(first)["evidence"][:1]
            out.append(
                {
                    "kind": "note",
                    "said": u["text"],
                    "said_at": fmt_time(u["start_s"]),
                    "id": first,
                    "in_view": [f"{sid}:{x['object_id']}" for x in links],
                    "evidence": ev,
                }
            )
    return out


def run(mem: Memory, sid: str, base_url: str) -> list[dict[str, Any]]:
    fs = facts(mem, sid)
    alerts: list[dict[str, Any]] = []
    if fs:
        out = chat_json(
            base_url, PROMPT.format(sid=sid, facts=json.dumps(fs, indent=1, ensure_ascii=False)), strength="medium"
        )
        for a in (out.get("alerts") if isinstance(out, dict) else None) or []:
            ans = {"answer": a.get("text", ""), "object_id": a.get("object_id"), "evidence": a.get("evidence", [])}
            problems = verify(ans, mem)
            if problems:
                log.info("dropped alert %r: %s", a.get("text"), problems)
                continue
            alerts.append(a)
    now = datetime.now().isoformat(timespec="seconds")
    with mem.con:
        mem.con.execute("DELETE FROM alerts WHERE session_id=?", (sid,))
        for a in alerts:
            o = mem._obj(a["object_id"])
            seen = (
                o["state_seen_at"]
                if o is not None and o["state_seen_at"] is not None
                else (o["last_seen"] if o is not None else None)
            )
            mem.con.execute(
                "INSERT INTO alerts VALUES (?,?,?,?,?,?,?)",
                (sid, a.get("kind"), a["text"], a["object_id"], json.dumps(a.get("evidence", [])), seen, now),
            )
    log.info("%s: %d facts -> %d alerts", sid, len(fs), len(alerts))
    return alerts
