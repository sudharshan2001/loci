"""Verifier: plain-code checks of an agent answer against the memory. Returns a list of problems."""

from __future__ import annotations

import json
import re
from typing import Any

from rapidfuzz import fuzz

from loci.agent.tools import Memory, fmt_time, split_uid

REL_RE = re.compile(r"\b(on|inside|left of|right of|near) the ([a-z][a-z \-]*?)(?=,|\.|;| in the |$)")
METRES_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:m\b|metres|meters)")
NEG_WORDS = re.compile(r"\b(off|not lit|unlit|no flame|closed|not on|not running|switched off)\b")
# "on" as a state, not "on your left" / "on the desk"
POS_WORDS = re.compile(
    r"\b(lit|on(?!\s+(?:the|your|a|an|my|top|left|right)\b)|burning|open|running|switched on|flame)\b"
)


def _labels(row) -> set[str]:
    return {row["label"], *json.loads(row["aliases"] or "[]")}


def _state_polarity(state: dict[str, Any]) -> bool | None:
    """The headline boolean of a stored state (first boolean field), e.g. burner_lit."""
    for v in state.values():
        if isinstance(v, bool):
            return v
    return None


def verify(ans: dict[str, Any], mem: Memory) -> list[str]:
    problems: list[str] = []
    text = (ans.get("answer") or "").lower()
    oid = ans.get("object_id")
    if not oid:
        # a summary across objects (e.g. "what changed"): every cited evidence id must exist
        for ev in ans.get("evidence") or []:
            ev_sid, _, obs = ev.rpartition(":")
            q = "SELECT 1 FROM observations WHERE obs_id=?" + (" AND session_id=?" if ev_sid else "")
            if mem.con.execute(q, (obs, ev_sid) if ev_sid else (obs,)).fetchone() is None:
                problems.append(f"evidence {ev} does not exist")
        return problems

    row = mem._obj(oid)
    if row is None:
        return [f"object_id {oid} does not exist"]
    sid, local = split_uid(oid)
    related = set(ans.get("objects") or []) | {oid}

    # evidence: exists, belongs to the named object (or a listed object), label matches
    if not ans.get("evidence"):
        problems.append("no evidence cited")
    for ev in ans.get("evidence") or []:
        # evidence ids may be capture-qualified ("<session>:obs_0033_01"); obs ids repeat across captures
        ev_sid, _, ev = ev.rpartition(":")
        if ev_sid and ev_sid != sid:
            problems.append(f"evidence {ev_sid}:{ev} is from capture {ev_sid}, but the object is in {sid}")
            continue
        ob = mem.con.execute("SELECT * FROM observations WHERE session_id=? AND obs_id=?", (sid, ev)).fetchone()
        if ob is None:
            problems.append(f"evidence {ev} does not exist in session {sid}")
            continue
        owner = f"{sid}:{ob['object_id']}"
        if owner not in related:
            problems.append(f"evidence {ev} belongs to {owner}, not {oid}")
        elif owner == oid and ob["label"] not in _labels(row):
            problems.append(f"evidence {ev} label {ob['label']!r} does not match {row['label']!r}")

    # position (filled by code, but check anyway): inside the object's box
    pos = ans.get("position_m")
    if pos is not None:
        lo, hi = json.loads(row["box_min"]), json.loads(row["box_max"])
        if not all(lo[i] - 0.05 <= pos[i] <= hi[i] + 0.05 for i in range(3)):
            problems.append("position_m is outside the object's 3D box")

    # relation words in location_text / answer must match stored relations
    # for a multi-object answer, a relation may describe any of the listed objects
    stored: set[tuple[str, str]] = set()
    known_texts: list[str] = []
    for u in related:
        r_sid, r_local = split_uid(u)
        for r in mem.con.execute(
            "SELECT r.relation, o.label FROM relations r JOIN objects o ON o.session_id=r.session_id "
            "AND o.object_id=r.object WHERE r.session_id=? AND r.subject=?",
            (r_sid, r_local),
        ):
            stored.add((r["relation"].replace("_of", " of").replace("_", " "), r["label"]))
        o = mem._obj(u)
        if o is not None and o["location_text"]:
            known_texts.append(o["location_text"].lower())
    # street places have no home relations; "on the right/left" there is a side, checked below
    sources = [] if row["distance_along_walk_m"] is not None else [(ans.get("location_text") or "").lower(), text]
    for src in sources:
        for word, target in REL_RE.findall(src):
            target = target.strip()
            if word == "on" and target.startswith(("your ", "left", "right", "the left", "the right", "burner")):
                continue  # a side or a part ("on the right burner"), not a thing it stands on
            if not any(word == w and fuzz.WRatio(target, lab) >= 85 for w, lab in stored):
                if any(f"{word} the {target}" in t for t in known_texts):
                    continue
                problems.append(f"relation '{word} the {target}' is not in the memory")

    # state claims: polarity must match, and must say when it was seen
    if row["state"]:
        state = json.loads(row["state"])
        pol = _state_polarity(state)
        said_neg, said_pos = bool(NEG_WORDS.search(text)), bool(POS_WORDS.search(text))
        if pol is True and said_neg and not said_pos:
            problems.append(f"answer says off/closed but stored state is {state}")
        if pol is False and said_pos and not said_neg:
            problems.append(f"answer says on/lit/open but stored state is {state}")
        seen = fmt_time(row["state_seen_at"])
        claims_state = said_neg or said_pos
        # any time mentioned within 2 s of when the state was seen counts (0:02 vs 0:00 is the same moment)
        said = [
            int(m) * 60 + int(sec) for m, sec in re.findall(r"\b(\d+):(\d{2})\b", f"{text} {ans.get('seen_at') or ''}")
        ]
        close = any(abs(t - row["state_seen_at"]) <= 2.0 for t in said)
        if claims_state and seen and not close:
            problems.append(f"state answer must say when it was seen ({seen})")
        if re.search(r"\b(right now|currently|is still|at the moment)\b", text):
            problems.append("state answers must be 'as of' when seen, not about now")

    # shop names: answer must contain the stored sign text (fuzzy)
    if row["text"]:
        names = [t for t in (row["text"], row["text_english"]) if t]
        if not any(fuzz.partial_ratio(n.lower(), text) >= 80 for n in names):
            problems.append(f"answer does not name the sign text {names[0]!r}")

    # street distance and side: for a list answer ("objects"), each mention must fit one of the listed places
    places = [r for r in (mem._obj(u) for u in related) if r is not None and r["distance_along_walk_m"] is not None]
    if places:
        for m in METRES_RE.findall(text):
            d = float(m)
            if not any(
                abs(d - r["distance_along_walk_m"]) <= max(10.0, 0.2 * r["distance_along_walk_m"]) for r in places
            ):
                refs = ", ".join(f"{r['distance_along_walk_m']:.0f}" for r in places)
                problems.append(f"distance {d:.0f} m does not match stored {refs} m")
        for side in ("left", "right"):
            if re.search(rf"\b(on your|to the|on the) {side}\b", text) and not any(r["side"] == side for r in places):
                problems.append(f"side '{side}' does not match stored side {places[0]['side']!r}")
    return problems
