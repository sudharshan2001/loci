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
    "street stall": [
        "stall",
        "stand",
        "kiosk",
        "fruit stand",
        "fruit stall",
        "vegetable stall",
        "coconut stall",
        "coconut",
        "shop front",
        "cart",
    ],
    "fruit stand": ["fruit stall", "fruit shop", "coconut stall", "coconut", "fruit"],
    "light": ["lights", "lamp", "tube light", "bulb", "ceiling light"],
    "laundry pile": ["laundry", "clothes", "washing"],
    "hut": ["brick hut", "shed", "shack", "brick building", "unfinished brick building", "cabin"],
    "chips": ["bag of chips", "snack", "snack packet", "packet", "food item", "crisps"],
    "bed": ["headboard", "bedsheet", "mattress", "bed frame"],
    "flower pot": ["pot", "planter", "plant pot", "potted plant"],
    "speaker": ["bluetooth speaker", "soundbar", "loudspeaker"],
    "garden": ["lawn", "balcony", "terrace", "yard"],
}

# one way: "my bag" may be a backpack, but "my purse" is not a backpack
KINDS_OF = {"bag": ["backpack", "rucksack", "school bag", "handbag", "purse", "tote bag", "duffel bag"]}
THROWAWAY = {"plastic bag", "paper bag"}  # nobody means these by "my bag": below real bags unless asked by name

COLOURS = {
    "red",
    "blue",
    "green",
    "yellow",
    "purple",
    "pink",
    "black",
    "white",
    "grey",
    "gray",
    "brown",
    "orange",
    "silver",
    "golden",
    "small",
    "big",
    "large",
    "my",
    "the",
}


def singular(word: str) -> str:
    """Crude English singular, enough for object names: shelves->shelf, pots->pot, boxes->box, batteries->battery."""
    w = word.lower()
    if w.endswith("ves") and len(w) > 4:
        return w[:-3] + "f"
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "sses")):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


DIRECT_MATCH = 101  # above any fuzzy score
LABEL_MATCH = 102  # the object's own name beats what vision said about it: a laptop "with a keyboard" is no keyboard


def yours(name: str | None) -> str | None:
    """Where a walk started, as said to the user: the owner's "my office" is "your office" in an answer."""
    return f"your {name[3:]}" if name and name.lower().startswith("my ") else name


def latin(text: str) -> bool:
    """Mostly Latin letters (English) rather than Tamil/Kannada script."""
    letters = [c for c in text if c.isalpha()]
    return sum(c.isascii() for c in letters) >= len(letters) / 2


def norm(phrase: str) -> str:
    return " ".join(singular(w) for w in phrase.lower().split())


def head_noun(phrase: str) -> str:
    words = [w for w in norm(phrase).split() if w not in COLOURS]
    return words[-1] if words else ""


# how long a sighting stays trustworthy, by how movable the thing is (hours; None = does not move)
MOVABILITY: list[tuple[tuple[str, ...], float | None]] = [
    (("keys", "key", "wallet", "phone", "charger", "glasses", "earphones", "remote", "passport", "watch"), 6.0),
    (("bag", "backpack", "bottle", "cup", "mug", "book", "notebook", "laptop", "umbrella", "plastic bag"), 12.0),
    (("laundry", "clothes", "jeans", "pillow", "blanket", "chair", "bucket", "box"), 24.0),
]


# what_changed: between two walks, positions of things that stayed put differ by up to ~2 m, and by up to ~3.5 m for
# things seen only briefly (measured on home/home2). A move needs a new thing to stand on, or a distance beyond that.
MOVED_MIN_M = 2.0
MOVED_ANYWAY_M = 4.0
SEEN_ONLY_SHOWN = 4
# a state change needs both sightings to be this sure; window frames filmed by day and by night read 0.65-0.75
STATE_MIN_CONFIDENCE = 0.7
# screens are blurred before anything looks at them, so on/off is not known from the frame
UNJUDGED_STATES = {"screen_on"}
# furniture and fixtures stay where they are, and walks name them differently ("kitchen shelf", "storage shelf")
STATIC_WORDS = {
    "desk", "table", "shelf", "shelving", "cabinet", "wardrobe", "bed", "sofa", "couch", "chair", "door", "window",
    "socket", "outlet", "switch", "stove", "fridge", "sink", "monitor", "curtain", "fan", "rack", "counter", "drawer",
}  # fmt: skip


def half_life_h(label: str) -> float | None:
    for words, h in MOVABILITY:
        if any(w == label or w in label.split() for w in words):
            return h
    return None


def _anchor(location_text: str | None) -> str:
    """The thing an object stands on or is inside, without the room: "on the desk, in the office" -> "on the desk"."""
    first = (location_text or "").split(",")[0].strip()
    return first if first.startswith(("on the ", "inside the ")) else ""


def _same_place(name: str, row: sqlite3.Row, other: str, other_row: sqlite3.Row) -> bool:
    """Two readings of one place on a walk: the same name close by, or a misreading right next to it on the same side
    ("Vangevar" 7 m from "Manyavar")."""
    gap = abs(row["distance_along_walk_m"] - other_row["distance_along_walk_m"])
    ratio = fuzz.ratio(name, other)
    if gap <= 40 and ratio >= 88:
        return True
    return gap <= 15 and ratio >= 60 and min(len(name), len(other)) >= 6 and row["side"] == other_row["side"]


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
    """The query plus synonyms. A single-word synonym only counts when the query is that one word
    (apart from colours/sizes), so "laptop stand" does not expand to street stalls."""
    q = q.lower().strip()
    nq = norm(q)
    core = [w for w in nq.split() if w not in COLOURS]
    terms = {q}
    for canon, syns in SYNONYMS.items():
        group = [canon, *syns]
        ngroup = {norm(g) for g in group}
        if nq in ngroup or (len(core) == 1 and core[0] in ngroup):
            terms |= set(group)
    if len(core) == 1:
        terms |= set(KINDS_OF.get(core[0], []))
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
        things_on = {
            (r["session_id"], r["object"]): r["n"]
            for r in self.con.execute(
                "SELECT session_id, object, COUNT(*) n FROM relations WHERE relation IN ('on','inside') "
                "GROUP BY session_id, object"
            )
        }
        asked = set(norm(query.strip()).split())
        kinds = {norm(t) for t in terms} & {norm(k) for ks in KINDS_OF.values() for k in ks}
        scored = []
        seen_as: dict[tuple[str, str], str] = {}
        same_kind: set[tuple[str, str]] = set()
        for r in rows:
            aliases = json.loads(r["aliases"] or "[]")
            names = [r["label"], *aliases]
            texts = [t for t in (r["text"], r["text_english"]) if t]
            # whole-word similarity: typos and "plant" vs "potted plant" match, "keys" vs "keyboard" does not;
            # aliases rank a little below the main label
            # subset matching ("plant" in "potted plant") only for the asked words, not for synonyms such as
            # "computer", which would otherwise match "computer mouse"
            ql = norm(query.strip())
            s_name = max(
                max(fuzz.ratio(norm(t), norm(n)), fuzz.token_set_ratio(norm(t), norm(n)) if norm(t) == ql else 0)
                - (0 if i == 0 else 5)
                for t in terms
                for i, n in enumerate(names)
            )
            # "purple bucket" should find "laundry bucket": when the other words are only colours/sizes,
            # the same head noun is a match ("laptop stand" must not match "fruit stand")
            only_colours = len([w for w in query.lower().split() if w not in COLOURS]) == 1
            hq = head_noun(query)
            if only_colours and any(head_noun(n) == hq for n in names):
                s_name = max(s_name, 85)
            # compound words: "shelf" finds "bookshelf"; "board" does not find "keyboard" (prefix "key" too short)
            elif (
                only_colours
                and len(hq) >= 5
                and any(w.endswith(hq) and len(w) - len(hq) >= 4 for n in names for w in norm(n).split())
            ):
                s_name = max(s_name, 82)
            s_text = max((fuzz.WRatio(query.lower(), t.lower()) for t in texts), default=0)
            score = max(s_name, s_text)
            # the words asked for are in what vision called this very object: above any synonym of its kind
            direct = next((a for a in aliases if asked and asked <= set(norm(a).split())), None)
            if norm(r["label"]) in kinds or (
                asked and asked <= set(norm(r["label"]).split()) and (r["label"] not in THROWAWAY or len(asked) > 1)
            ):
                score = LABEL_MATCH
            elif direct:
                score = DIRECT_MATCH
                seen_as[(r["session_id"], r["object_id"])] = direct
                if max(fuzz.ratio(norm(t), norm(r["label"])) for t in terms) >= 80:
                    same_kind.add((r["session_id"], r["object_id"]))
            if score >= 80:
                scored.append((score, r))
        if any(sc == LABEL_MATCH for sc, _ in scored):
            scored = [
                (sc, r) for sc, r in scored if sc != DIRECT_MATCH or (r["session_id"], r["object_id"]) in same_kind
            ]
        scored.sort(
            key=lambda x: (-x[0], -things_on.get((x[1]["session_id"], x[1]["object_id"]), 0), -x[1]["last_seen"])
        )
        # at most 3 per label in each capture; labels then captures take turns, so a related kind ("fruit stand"
        # for "coconut stall") or a short capture is not crowded out by a long walk with many objects of one label
        per_label: dict[tuple[str, str], int] = {}
        per_session: dict[str, int] = {}
        varied = []
        for sc, r in scored:
            key = (r["session_id"], r["label"])
            label_turn = per_label.get(key, 0)
            per_label[key] = label_turn + 1
            if label_turn < 3:
                session_turn = per_session.get(r["session_id"], 0)
                per_session[r["session_id"]] = session_turn + 1
                varied.append((label_turn, session_turn, sc, r))
        varied.sort(key=lambda x: (-max(x[2], DIRECT_MATCH - 1), x[0], x[1], -x[2], -x[3]["last_seen"]))
        scored = [(sc, r) for _, _, sc, r in varied]
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
                **({"seen_as": a} if (a := seen_as.get((r["session_id"], r["object_id"]))) else {}),
                **({"things_on": n} if (n := things_on.get((r["session_id"], r["object_id"]))) else {}),
                # "did I leave the fan on?" with two fans: each one's state, so one seen off does not hide one left on
                **(
                    {"state": st, "state_seen_at": fmt_time(r["state_seen_at"])}
                    if r["state"] and (st := json.loads(r["state"]))
                    else {}
                ),
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

    def _room_pos(self, row: sqlite3.Row) -> np.ndarray | None:
        """Object position in the shared room frame, if its capture is aligned to it."""
        from loci.register import apply

        r = self.con.execute("SELECT to_room_frame FROM sessions WHERE session_id=?", (row["session_id"],)).fetchone()
        if not r or not r["to_room_frame"] or row["cx"] is None:
            return None
        return apply(json.loads(r["to_room_frame"])["transform"], [row["cx"], row["cy"], row["cz"]])

    def last_known(self, query: str, now: str | None = None) -> list[dict[str, Any]]:
        """The same thing across captures (linked in the room frame): sightings over time, whether it moved,
        and how likely it is still there given how movable it is. Newest sighting first."""
        from datetime import datetime

        hits = self.search_objects(query, limit=30)
        things: list[dict[str, Any]] = []
        for h in hits:
            row = self._obj(h["id"])
            pos = self._room_pos(row)
            captured = self._captured(row["session_id"]) or ""
            sight = {
                "id": h["id"],
                "capture": row["session_id"],
                "captured_at": captured,
                "seen_at": fmt_time(row["last_seen"]),
                "where": row["location_text"],
                "evidence": [row["best_obs"]],
                "room_pos": None if pos is None else pos.round(2).tolist(),
            }
            # link to an existing thing with the same label within 0.6 m in the room frame
            match = None
            if pos is not None:
                for t in things:
                    if t["label"] == row["label"] and t["_pos"] is not None and np.linalg.norm(t["_pos"] - pos) < 0.6:
                        match = t
                        break
            if match is None:
                match = {"label": row["label"], "_pos": pos, "sightings": []}
                things.append(match)
            match["sightings"].append(sight)
        ref = datetime.fromisoformat(now) if now else datetime.now().astimezone()
        out = []
        for t in things:
            sights = sorted(t["sightings"], key=lambda x: x["captured_at"], reverse=True)
            latest = sights[0]
            hl = half_life_h(t["label"])
            p_there = None
            if latest["captured_at"]:
                age_h = max(0.0, (ref - datetime.fromisoformat(latest["captured_at"])).total_seconds() / 3600)
                p_there = 1.0 if hl is None else round(0.5 ** (age_h / hl), 2)
            moved = [
                s
                for s in sights[1:]
                if s["room_pos"]
                and latest["room_pos"]
                and np.linalg.norm(np.subtract(s["room_pos"], latest["room_pos"])) > 0.3
            ]
            out.append(
                {
                    "label": t["label"],
                    "latest": latest,
                    "earlier": sights[1:],
                    "times_seen": len(sights),
                    "movable": hl is not None,
                    "still_there_probability": p_there,
                    "moved_between_captures": bool(moved),
                }
            )
        return out

    def list_captures(self) -> list[dict[str, Any]]:
        rows = self.con.execute(
            "SELECT session_id, scene, captured_at, start_name FROM sessions ORDER BY captured_at"
        ).fetchall()
        return [{**dict(r), "start_name": yours(r["start_name"])} for r in rows]

    def _pairs(
        self, ra: list[sqlite3.Row], rb: list[sqlite3.Row]
    ) -> list[tuple[sqlite3.Row, sqlite3.Row, float | None]]:
        """Same-label objects of two captures, one to one: closest first in the room frame, else in the order seen."""
        pa, pb = [self._room_pos(r) for r in ra], [self._room_pos(r) for r in rb]
        if any(p is None for p in pa + pb):
            return [(x, y, None) for x, y in zip(ra, rb, strict=False)]
        dist = sorted((float(np.linalg.norm(p - q)), i, j) for i, p in enumerate(pa) for j, q in enumerate(pb))
        pairs: list[tuple[sqlite3.Row, sqlite3.Row, float | None]] = []
        used_a: set[int] = set()
        used_b: set[int] = set()
        for d, i, j in dist:
            if i not in used_a and j not in used_b:
                used_a.add(i)
                used_b.add(j)
                pairs.append((ra[i], rb[j], d))
        return pairs

    def what_changed(self, session_a: str, session_b: str) -> dict[str, Any]:
        """Differences from capture a (earlier) to capture b (later): state changes, things moved, things seen on
        only one walk.

        Objects are matched by label (and sign text), closest first in the room frame. Furniture is never reported:
        its position error between walks is larger than any real move. Something seen on one walk only may simply
        not have been filmed on the other, so those lists say that and nothing more.
        """

        def by_key(sid: str) -> dict[str, list[sqlite3.Row]]:
            out: dict[str, list[sqlite3.Row]] = {}
            for r in self.con.execute("SELECT * FROM objects WHERE session_id=? ORDER BY first_seen", (sid,)):
                out.setdefault(r["text"] or r["label"], []).append(r)
            return out

        def sure(r: sqlite3.Row) -> bool:
            return r["attribute_confidence"] is None or r["attribute_confidence"] >= STATE_MIN_CONFIDENCE

        a, b = by_key(session_a), by_key(session_b)
        if not a or not b:
            return {"error": f"no objects in {session_a if not a else session_b}"}
        res: dict[str, Any] = {
            "from": session_a,
            "to": session_b,
            "from_captured_at": self._captured(session_a),
            "to_captured_at": self._captured(session_b),
            # most telling first: long lists can run past what the agent is shown
            "state_changed": [],
            "moved": [],
            "seen_only_earlier": [],
            "seen_only_later": [],
            "note": "seen_only lists may just not have been filmed on the other walk; they are not gone or new",
        }
        for key in sorted(set(a) | set(b)):
            ra, rb = a.get(key, []), b.get(key, [])
            static = bool(STATIC_WORDS & set(key.split()))
            if not ra or not rb:
                if not static:
                    r = (ra or rb)[0]
                    side = "seen_only_earlier" if ra else "seen_only_later"
                    res[side].append({"id": uid(r), "label": key, "where": r["location_text"]})
                continue
            for x, y, d in self._pairs(ra, rb):
                was, now = _anchor(x["location_text"]), _anchor(y["location_text"])
                if d is None:  # captures not aligned: only the thing it stands on can tell
                    shifted = bool(was and now and was != now)
                else:
                    shifted = d >= MOVED_ANYWAY_M or (d >= MOVED_MIN_M and bool(was and now and was != now))
                if shifted and not static:
                    res["moved"].append(
                        {
                            "id": uid(y),
                            "label": key,
                            **({} if d is None else {"moved_m": round(d, 2)}),
                            "was": x["location_text"],
                            "now": y["location_text"],
                            "evidence": [y["best_obs"]],
                        }
                    )
                only_one = len(ra) == len(rb) == 1
                same_thing = only_one or (d is not None and d < MOVED_MIN_M)
                if not (same_thing and x["state"] and y["state"] and sure(x) and sure(y)):
                    continue
                sx, sy = json.loads(x["state"]), json.loads(y["state"])
                if sx != sy and not UNJUDGED_STATES & set(sy):
                    res["state_changed"].append(
                        {
                            "id": uid(y),
                            "label": key,
                            "was": sx,
                            "was_seen_at": fmt_time(x["state_seen_at"]),
                            "now": sy,
                            "now_seen_at": fmt_time(y["state_seen_at"]),
                            "evidence": [y["best_obs"]],
                        }
                    )
        for side in ("seen_only_earlier", "seen_only_later"):  # things that move first; the answer names only a few
            res[side] = sorted(res[side], key=lambda t: half_life_h(t["label"]) is None)[:SEEN_ONLY_SHOWN]
        return res

    def route(self, from_id: str, to_id: str) -> dict[str, Any]:
        """Walking directions between two things seen on the same walk, following the path you took.

        from_id may be "start": the beginning of that walk (e.g. the user's office)."""
        if from_id == "start":
            b = self._obj(to_id)
            if b is None or b["distance_along_walk_m"] is None:
                return {"error": f"unknown street object {to_id}"}
            start = self.con.execute(
                "SELECT start_name FROM sessions WHERE session_id=?", (b["session_id"],)
            ).fetchone()
            return {
                "from": yours(start["start_name"]) or "the start of the walk",
                "to": to_id,
                "walk_m": b["distance_along_walk_m"],
                "direction": "the way you walked",
                "destination_side": f"on your {b['side']}",
                "passed_at": fmt_time(b["passed_at"]),
                "evidence": [b["best_obs"]],
            }
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
            "AND ((text IS NOT NULL AND text_clear IS NOT 0) OR label LIKE '%stall%' OR label LIKE '%shop%') "
            "ORDER BY distance_along_walk_m",
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

    def places_passed(self, session_id: str | None = None, per_walk: int = 40) -> list[dict[str, Any]]:
        """Places along the walks, in walk order. Only clearly read names. The same name read again nearby counts as one
        place, called what it was read as most, and a long walk keeps the places seen most, so the list fits."""
        # read signs, plus stalls and shop fronts even when no sign text could be read
        if not session_id:  # no walk named: the newest one, never a mix of walks
            newest = self.con.execute(
                "SELECT session_id FROM sessions WHERE scene='street' ORDER BY captured_at DESC LIMIT 1"
            ).fetchone()
            session_id = newest["session_id"] if newest else None
        q = (
            "SELECT * FROM objects WHERE distance_along_walk_m IS NOT NULL "
            "AND ((text IS NOT NULL AND text != '' AND text_clear IS NOT 0) "
            "OR label LIKE '%stall%' OR label LIKE '%stand%' OR label LIKE '%shop front%' OR label LIKE '%kiosk%')"
        )
        args: tuple = ()
        if session_id:
            q += " AND session_id=?"
            args = (session_id,)
        rows = self.con.execute(q + " ORDER BY session_id, distance_along_walk_m", args).fetchall()
        walks: dict[str, list[dict[str, Any]]] = {}
        for r in rows:
            name = (r["text_english"] or r["text"] or "").lower().strip()
            places = walks.setdefault(r["session_id"], [])
            same = next((p for p in places if name and _same_place(name, r, p["name"], p["row"])), None)
            if same:
                same["seen"] += r["n_obs"] or 1
                if (r["n_obs"] or 1) > (same["row"]["n_obs"] or 1):
                    same.update(row=r, name=name)
            else:
                places.append({"name": name, "row": r, "seen": r["n_obs"] or 1})
        out = []
        for places in walks.values():
            for p in places:
                # one spelling for a name read at several shops of a walk: "Alien Solly" is the rarer "Allen Solly"
                likely = max(
                    (q for q in places if q["name"] and fuzz.ratio(p["name"], q["name"]) >= 88),
                    key=lambda q: q["seen"],
                    default=p,
                )
                p["spelled"] = likely["row"]
            # unnamed shop fronts only fill what is left: a short walk has stalls with no sign, a long one has names
            top = sorted(places, key=lambda p: (not p["name"], -p["seen"]))[:per_walk]
            out += sorted(top, key=lambda p: p["row"]["distance_along_walk_m"])
        return [
            {
                "id": uid(r),
                "name": p["spelled"]["text"] or r["text"] or r["label"],
                "name_english": p["spelled"]["text_english"] or r["text_english"],
                "open": json.loads(r["state"]).get("shop_open") if r["state"] else None,
                "distance_along_walk_m": r["distance_along_walk_m"],
                "side": r["side"],
                "passed_at": fmt_time(r["passed_at"]),
                "evidence": [r["best_obs"]],
            }
            for p, r in ((p, p["row"]) for p in out)
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
            # "trash" never fuzzy-matches "குப்பை": a note in another script is offered for the agent to read itself
            other_script = latin(text) != latin(u["text"])
            if max(score, label_hit) >= 60 or other_script:
                out.append(
                    {
                        "session": u["session_id"],
                        "said_at": fmt_time(u["start_s"]),
                        "text": u["text"],
                        "objects_in_view": [f"{u['session_id']}:{x['object_id']}" for x in links[:5]],
                        "match": max(score, label_hit),
                        **({"note": "said in another language: read it and judge"} if other_script else {}),
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
