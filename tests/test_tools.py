import json
import sqlite3

from loci.agent.tools import Memory
from loci.stages.store import SCHEMA


def _mem() -> Memory:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.execute("INSERT INTO sessions (session_id, scene, captured_at) VALUES ('am', 'home', '2026-10-28T09:00:00')")
    con.execute("INSERT INTO sessions (session_id, scene, captured_at) VALUES ('pm', 'home', '2026-10-28T18:40:00')")
    con.execute(
        "INSERT INTO sessions (session_id, scene, captured_at) VALUES ('walk', 'street', '2026-10-28T17:00:00')"
    )

    def obj(sid, oid, label, **kw):
        cols = {
            "object_id": oid,
            "session_id": sid,
            "label": label,
            "aliases": "[]",
            "cx": 0,
            "cy": 0,
            "cz": 0,
            "last_seen": 5.0,
            "confidence": 0.9,
            "best_obs": f"obs_{oid}",
            **kw,
        }
        con.execute(f"INSERT INTO objects ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})", list(cols.values()))
        con.execute(
            "INSERT INTO observations (obs_id, session_id, object_id, t) VALUES (?,?,?,?)",
            (f"obs_{oid}", sid, oid, kw.get("state_seen_at", 5.0)),
        )

    obj("am", "gas_stove_1", "gas stove", state=json.dumps({"burner_lit": False}), state_seen_at=3.0)
    obj("pm", "gas_stove_1", "gas stove", state=json.dumps({"burner_lit": True}), state_seen_at=31.0)
    obj("walk", "coconut_stall_1", "coconut stall", distance_along_walk_m=2.0, side="left")
    obj("walk", "shop_sign_1", "shop sign", text="Sri Balaji Stores", distance_along_walk_m=9.0, side="right")
    obj("walk", "fruit_stall_1", "fruit stall", distance_along_walk_m=20.0, side="right")
    return Memory(con)


def test_state_history_newest_first() -> None:
    hist = _mem().state_history("stove")
    assert [h["capture"] for h in hist] == ["pm", "am"]
    assert hist[0]["state"] == {"burner_lit": True} and hist[0]["seen_at"] == "0:31"


def test_route_forward_and_back() -> None:
    mem = _mem()
    fwd = mem.route("walk:coconut_stall_1", "walk:fruit_stall_1")
    assert fwd["walk_m"] == 18.0 and fwd["direction"] == "same way you walked"
    assert fwd["destination_side"] == "on your right"
    assert [p["name"] for p in fwd["you_pass"]] == ["Sri Balaji Stores"]
    back = mem.route("walk:fruit_stall_1", "walk:coconut_stall_1")
    assert back["direction"] == "back the way you came" and back["destination_side"] == "on your right"


def test_route_rejects_different_captures() -> None:
    assert "error" in _mem().route("am:gas_stove_1", "walk:fruit_stall_1")


def test_what_changed_states_moves_added_removed() -> None:
    mem = _mem()
    con = mem.con
    for sid, oid, label, where in [
        ("am", "keys_1", "keys", "on the desk"),
        ("pm", "keys_1", "keys", "on the bed"),
        ("am", "laundry_pile_1", "laundry pile", "near the door"),
        ("pm", "phone_1", "phone", "on the desk"),
    ]:
        con.execute(
            "INSERT INTO objects (object_id, session_id, label, aliases, location_text, best_obs) VALUES (?,?,?,?,?,?)",
            (oid, sid, label, "[]", where, f"obs_{oid}"),
        )
    ch = mem.what_changed("am", "pm")
    assert [c["label"] for c in ch["state_changed"]] == ["gas stove"]
    assert ch["state_changed"][0]["was"] == {"burner_lit": False} and ch["state_changed"][0]["now"] == {
        "burner_lit": True
    }
    assert [(m["label"], m["was"], m["now"]) for m in ch["moved"]] == [("keys", "on the desk", "on the bed")]
    assert [x["label"] for x in ch["added"]] == ["phone"]
    assert [x["label"] for x in ch["removed"]] == ["laundry pile"]


def test_search_head_noun_typos_and_no_false_keys() -> None:
    mem = _mem()
    for oid, label in [
        ("laundry_bucket_1", "laundry bucket"),
        ("keyboard_1", "keyboard"),
        ("potted_plant_1", "potted plant"),
    ]:
        mem.con.execute(
            "INSERT INTO objects (object_id, session_id, label, aliases, last_seen) VALUES (?,?,?,?,?)",
            (oid, "am", label, "[]", 1.0),
        )
    mem.con.execute(
        "INSERT INTO objects (object_id, session_id, label, aliases, text, last_seen) VALUES (?,?,?,?,?,?)",
        ("signboard_1", "walk", "signboard", "[]", "Sudarshan Silks", 1.0),
    )
    labels = lambda q: [h["label"] for h in mem.search_objects(q)]  # noqa: E731
    assert labels("purple bucket") == ["laundry bucket"]
    assert labels("keys") == []  # keyboard keys are not keys
    assert labels("plants") == ["potted plant"]
    assert "signboard" in labels("Sudharshan Silks")
    assert "fruit stall" in labels("coconut stall")


def test_last_known_links_across_aligned_captures_and_decays() -> None:
    import numpy as np

    mem = _mem()
    # evening capture frame = morning frame shifted by +1 m in x
    T = np.eye(4)
    T[0, 3] = -1.0  # evening -> room (morning) frame
    mem.con.execute(
        "UPDATE sessions SET to_room_frame=? WHERE session_id='am'", (json.dumps({"transform": np.eye(4).tolist()}),)
    )
    mem.con.execute(
        "UPDATE sessions SET to_room_frame=? WHERE session_id='pm'", (json.dumps({"transform": T.tolist()}),)
    )
    rows = [
        ("am", "keys_1", "keys", [0.0, 0.0, 0.0], "on the desk"),
        ("pm", "keys_1", "keys", [3.0, 0.0, 0.0], "on the bed"),  # room x = 2.0: moved 2 m
        ("am", "desk_1", "desk", [0.5, 0.0, 0.0], None),
        ("pm", "desk_1", "desk", [1.5, 0.0, 0.0], None),  # room x = 0.5: same desk
    ]
    for sid, oid, label, c, where in rows:
        mem.con.execute(
            "INSERT INTO objects (object_id, session_id, label, aliases, cx, cy, cz, last_seen, location_text,"
            " best_obs) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (oid, sid, label, "[]", *c, 4.0, where, f"obs_{oid}"),
        )
    keys = mem.last_known("keys", now="2026-10-29T00:40:00")
    assert len(keys) == 2 and not any(k["moved_between_captures"] for k in keys)  # 2 m apart: two places
    desk = mem.last_known("desk", now="2026-10-29T00:40:00")
    assert len(desk) == 1 and desk[0]["times_seen"] == 2 and desk[0]["still_there_probability"] == 1.0
    latest_keys = max(keys, key=lambda k: k["latest"]["captured_at"])
    assert (
        latest_keys["latest"]["where"] == "on the bed" and latest_keys["still_there_probability"] == 0.5
    )  # 6 h later, keys half-life 6 h
    ch = mem.what_changed("am", "pm")
    assert ("keys", 2.0) in [(m["label"], m["moved_m"]) for m in ch["moved"]]


def test_plurals_compounds_and_new_synonyms() -> None:
    from loci.agent.tools import expand_query, norm, singular

    assert [singular(w) for w in ["shelves", "pots", "boxes", "batteries", "glass", "keys"]] == [
        "shelf",
        "pot",
        "box",
        "battery",
        "glass",
        "key",
    ]
    assert norm("White Pots") == "white pot"
    assert "unfinished brick building" in expand_query("brick hut")
    assert "food item" in expand_query("chips")
    mem = _mem()
    for oid, label in [
        ("wooden_bookshelf_1", "wooden bookshelf"),
        ("white_flower_pot_1", "white flower pot"),
        ("keyboard_1", "keyboard"),
    ]:
        mem.con.execute(
            "INSERT INTO objects (object_id, session_id, label, aliases, last_seen) VALUES (?,?,?,?,?)",
            (oid, "am", label, "[]", 1.0),
        )
    labels = lambda q: [h["label"] for h in mem.search_objects(q)]  # noqa: E731
    assert labels("shelves") == ["wooden bookshelf"]
    assert labels("white pots") == ["white flower pot"]
    assert "keyboard" not in labels("board")  # "key" is too short a prefix for a compound match
