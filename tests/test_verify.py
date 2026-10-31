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


def test_negated_state_contradicts_a_lit_burner(mem: Memory) -> None:
    ans = {"object_id": "home:gas_stove_1", "evidence": ["obs_gas_stove_1"], "seen_at": "0:31"}
    assert any("stored state" in p for p in verify({**ans, "answer": "As of 0:31 the burner was not lit."}, mem))
    assert any("stored state" in p for p in verify({**ans, "answer": "No, the stove was unattended at 0:31."}, mem))
    assert verify({**ans, "answer": "As of 0:31 the burner was lit."}, mem) == []


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


def test_a_walk_is_not_something_things_stand_on(mem: Memory) -> None:
    ans = {
        "answer": "Your keys are on the desk; they were only seen on the other walk.",
        "object_id": "home:keys_1",
        "evidence": ["obs_keys_1"],
        "location_text": "on the desk, in the bedroom",
    }
    assert verify(ans, mem) == []


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


def test_state_time_within_two_seconds_is_fine(mem: Memory) -> None:
    ans = {"answer": "As of 0:32 the burner was lit.", "object_id": "home:gas_stove_1", "evidence": ["obs_gas_stove_1"]}
    assert verify(ans, mem) == []
    far = {**ans, "answer": "As of 0:40 the burner was lit."}
    assert any("when it was seen" in p for p in verify(far, mem))


def test_evidence_from_another_capture_is_rejected(mem: Memory) -> None:
    from loci.agent.agent import qualify_evidence

    ok = {
        "answer": "Your keys are on the desk.",
        "object_id": "home:keys_1",
        "evidence": ["home:obs_keys_1"],
        "location_text": "on the desk, in the bedroom",
    }
    assert verify(ok, mem) == []
    bad = {**ok, "evidence": ["street:obs_keys_1"]}
    assert any("from capture street" in p for p in verify(bad, mem))
    res = qualify_evidence([{"id": "home:tv_1", "evidence": ["obs_1"]}, {"capture": "pm", "evidence": ["obs_2"]}])
    assert res == [{"id": "home:tv_1", "evidence": ["home:obs_1"]}, {"capture": "pm", "evidence": ["pm:obs_2"]}]


def test_comparing_walks_may_cite_the_same_thing_in_the_other_walk(mem: Memory) -> None:
    mem.con.execute("INSERT INTO sessions (session_id, scene) VALUES ('later', 'home')")
    mem.con.execute(
        "INSERT INTO observations (obs_id, session_id, object_id, keyframe, t, label, score) "
        "VALUES ('obs_9', 'later', 'gas_stove_9', 0, 1.0, 'gas stove', 0.9)"
    )
    ans = {
        "answer": "As of 0:31 the burner was lit.",
        "object_id": "home:gas_stove_1",
        "evidence": ["home:obs_gas_stove_1", "later:obs_9"],
        "seen_at": "0:31",
    }
    assert verify(ans, mem) == []
    assert any("no evidence from capture home" in p for p in verify({**ans, "evidence": ["later:obs_9"]}, mem))
    # another kind of thing in the other walk is still a wrong pairing
    keys = {**ans, "object_id": "home:keys_1", "answer": "Your keys.", "evidence": ["home:obs_keys_1", "later:obs_9"]}
    assert any("from capture later" in p for p in verify(keys, mem))


def test_multi_object_answers(mem: Memory) -> None:
    # a change summary without a single object: evidence must exist
    assert verify({"answer": "Your keys moved.", "object_id": None, "evidence": ["home:obs_keys_1"]}, mem) == []
    assert verify({"answer": "x", "object_id": None, "evidence": ["home:obs_nope"]}, mem)
    # a relation describing another listed object is fine
    ans = {
        "answer": "The stove was lit; your keys are on the desk.",
        "object_id": "home:gas_stove_1",
        "objects": ["home:keys_1"],
        "evidence": ["home:obs_gas_stove_1"],
        "seen_at": "0:31",
    }
    assert verify(ans, mem) == []


def test_times_in_the_answer_must_be_when_it_was_seen(mem: Memory) -> None:
    good = {
        "answer": "Sudarshan Silks was on your right at about 0:30.",
        "object_id": "street:shop_sign_1",
        "evidence": ["obs_shop_sign_1"],
    }
    assert verify(good, mem) == []
    bad = {**good, "answer": "ಸುದರ್ಶನ್ Sudarshan Silks 1:41 ಕ್ಕೆ."}
    assert any("1:41" in p for p in verify(bad, mem))
    assert any("1:41" in p for p in verify({**good, "seen_at": "1:41"}, mem))


def test_times_may_come_from_other_things_named(mem: Memory) -> None:
    mem.con.execute("UPDATE objects SET first_seen=50, last_seen=60 WHERE object_id='desk_1'")
    ans = {
        "answer": "Your keys were seen at 0:31 and the desk at 0:55.",
        "object_id": "home:keys_1",
        "evidence": ["obs_keys_1"],
    }
    assert verify(ans, mem) == []
    # a clock time on a sign is not a capture time
    assert not [p for p in verify({**ans, "answer": "Your keys, seen at 0:31, open at 10:30 am."}, mem) if "10:30" in p]
