import json
import sqlite3

import pytest

from loci.agent.tools import Memory
from loci.agent.verify import verify
from loci.stages.store import SCHEMA


def _obj(con, sid, oid, label, c, **kw):
    cols = {
        "object_id": oid,
        "session_id": sid,
        "label": label,
        "aliases": "[]",
        "area": kw.get("area", "kitchen"),
        "cx": c[0],
        "cy": c[1],
        "cz": c[2],
        "box_min": json.dumps([x - 0.1 for x in c]),
        "box_max": json.dumps([x + 0.1 for x in c]),
        "extent_m": "[0.2,0.2,0.2]",
        "first_seen": 1.0,
        "last_seen": kw.get("last_seen", 31.0),
        "n_obs": 3,
        "confidence": 0.9,
        "best_obs": f"obs_{oid}",
        "location_text": kw.get("location_text"),
        "text": kw.get("text"),
        "text_english": kw.get("text_english"),
        "state": json.dumps(kw["state"]) if "state" in kw else None,
        "state_seen_at": kw.get("state_seen_at"),
        "distance_along_walk_m": kw.get("dist"),
        "side": kw.get("side"),
    }
    con.execute(f"INSERT INTO objects ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", list(cols.values()))
    con.execute(
        "INSERT INTO observations (obs_id, session_id, object_id, keyframe, t, label, score) VALUES (?,?,?,?,?,?,?)",
        (f"obs_{oid}", sid, oid, 0, 1.0, label, 0.9),
    )


@pytest.fixture()
def mem() -> Memory:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.execute("INSERT INTO sessions (session_id, scene) VALUES ('home', 'home'), ('street', 'street')")
    _obj(con, "home", "gas_stove_1", "gas stove", [0, 0, 1], state={"burner_lit": True}, state_seen_at=31.0)
    _obj(con, "home", "desk_1", "desk", [2, 0, 2])
    _obj(con, "home", "keys_1", "keys", [2, -0.4, 2], location_text="on the desk, in the bedroom")
    con.execute("INSERT INTO relations VALUES ('home', 'keys_1', 'on', 'desk_1', 0.4)")
    _obj(con, "street", "shop_sign_1", "shop sign", [5, 0, 80], text="Sudarshan Silks", dist=80.0, side="right")
    return Memory(con)


def test_good_state_answer(mem: Memory) -> None:
    ans = {
        "answer": "When you left the kitchen at 0:31, the burner was lit.",
        "object_id": "home:gas_stove_1",
        "evidence": ["obs_gas_stove_1"],
        "seen_at": "0:31",
    }
    assert verify(ans, mem) == []


def test_state_contradiction_and_now_claim(mem: Memory) -> None:
    ans = {
        "answer": "The stove is off right now.",
        "object_id": "home:gas_stove_1",
        "evidence": ["obs_gas_stove_1"],
    }
    problems = verify(ans, mem)
    assert any("stored state" in p for p in problems)
    assert any("not about now" in p for p in problems)
    assert any("when it was seen" in p for p in problems)


def test_wrong_evidence_and_missing_object(mem: Memory) -> None:
    assert verify({"answer": "x", "object_id": "home:nope_1", "evidence": []}, mem)[0].startswith("object_id")
    ans = {"answer": "Your keys are on the desk.", "object_id": "home:keys_1", "evidence": ["obs_desk_1"]}
    assert any("belongs to" in p for p in verify(ans, mem))


def test_relation_check(mem: Memory) -> None:
    good = {
        "answer": "Your keys are on the desk.",
        "object_id": "home:keys_1",
        "evidence": ["obs_keys_1"],
        "location_text": "on the desk, in the bedroom",
    }
    assert verify(good, mem) == []
    bad = {**good, "answer": "Your keys are inside the drawer."}
    assert any("inside the drawer" in p for p in verify(bad, mem))


def test_shop_name_distance_side(mem: Memory) -> None:
    good = {
        "answer": "Sudharshan Silks was on your right, about 80 m into the walk.",
        "object_id": "street:shop_sign_1",
        "evidence": ["obs_shop_sign_1"],
    }
    assert verify(good, mem) == []
    bad = {**good, "answer": "Sudarshan Silks was on your left, about 200 m into the walk."}
    problems = verify(bad, mem)
    assert any("distance" in p for p in problems) and any("side" in p for p in problems)


def test_refusal_passes(mem: Memory) -> None:
    assert verify({"answer": "I haven't seen a passport in any capture.", "object_id": None}, mem) == []


def test_list_answer_distances_match_listed_places(mem: Memory) -> None:
    mem.con.execute(
        "INSERT INTO objects (object_id, session_id, label, aliases, box_min, box_max, distance_along_walk_m, side,"
        " text, best_obs) VALUES ('shop_sign_2','street','shop sign','[]','[0,0,0]','[1,1,1]',30.0,'left',"
        "'Fresh Fruits','obs_shop_sign_2')"
    )
    ans = {
        "answer": "You passed Fresh Fruits on your left at about 30 m, then Sudarshan Silks on your right at 80 m.",
        "object_id": "street:shop_sign_1",
        "objects": ["street:shop_sign_2"],
        "evidence": ["obs_shop_sign_1"],
    }
    assert verify(ans, mem) == []
    bad = {**ans, "answer": "You passed Fresh Fruits at 150 m and Sudarshan Silks on your right at 80 m."}
    assert any("150" in p for p in verify(bad, mem))


def test_street_sides_are_not_home_relations(mem: Memory) -> None:
    ans = {
        "answer": "On the street walk you passed Sudarshan Silks on the right, about 80 m in.",
        "object_id": "street:shop_sign_1",
        "evidence": ["obs_shop_sign_1"],
    }
    assert verify(ans, mem) == []
