import json
import sqlite3

import numpy as np

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
    assert [x["label"] for x in ch["seen_only_later"]] == ["phone"]
    assert [x["label"] for x in ch["seen_only_earlier"]] == ["laundry pile"]


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


def _aligned(rows: list[tuple]) -> Memory:
    mem = _mem()
    mem.con.execute("DELETE FROM objects")
    frame = json.dumps({"transform": np.eye(4).tolist()})
    mem.con.execute("UPDATE sessions SET to_room_frame=? WHERE session_id IN ('am', 'pm')", (frame,))
    for sid, oid, label, x, where, *rest in rows:
        state, conf = (rest + [None, None])[:2]
        mem.con.execute(
            "INSERT INTO objects (object_id, session_id, label, aliases, cx, cy, cz, location_text, best_obs, state,"
            " attribute_confidence) VALUES (?,?,?,'[]',?,0,0,?,?,?,?)",
            (oid, sid, label, x, where, f"obs_{oid}", state, conf),
        )
    return mem


def test_what_changed_does_not_report_position_noise() -> None:
    ch = _aligned(
        [
            ("am", "desk_1", "desk", 0.0, "in the office"),
            ("pm", "desk_1", "desk", 5.0, "in the kitchen"),  # furniture: never reported, however far
            ("am", "laptop_1", "laptop", 0.0, "on the desk, in the office"),
            ("pm", "laptop_1", "laptop", 3.0, "on the desk, in the kitchen"),  # same place, room name differs
            ("am", "bag_1", "bag", 0.0, "on the bed"),
            ("pm", "bag_1", "bag", 1.5, "on the desk"),  # inside the position error
            ("am", "mug_1", "mug", 0.0, "on the desk, in the office"),
            ("pm", "mug_1", "mug", 3.5, "in the bedroom"),  # no new surface, and within the error for brief sightings
            ("am", "shelf_1", "kitchen shelf", 0.0, None),
            ("pm", "air_fryer_1", "air fryer", 0.0, "on the desk"),
        ]
    ).what_changed("am", "pm")
    assert ch["moved"] == [] and ch["seen_only_earlier"] == []
    assert [x["label"] for x in ch["seen_only_later"]] == ["air fryer"]


def test_what_changed_pairs_nearest_and_needs_sure_states() -> None:
    on, off = '{"burner_lit": true}', '{"burner_lit": false}'
    op, shut = '{"window_open": true}', '{"window_open": false}'
    ch = _aligned(
        [
            ("am", "keys_1", "keys", 1.0, "on the desk"),
            ("am", "keys_2", "keys", 9.0, "on the bed"),
            ("pm", "keys_1", "keys", 9.1, "on the bed"),  # order differs: still the same two sets of keys
            ("pm", "keys_2", "keys", 1.1, "on the desk"),
            ("am", "gas_stove_1", "gas stove", 0.0, None, on, 0.75),
            ("pm", "gas_stove_1", "gas stove", 0.2, None, off, 0.85),
            ("am", "window_1", "window", 0.0, None, op, 0.75),
            ("pm", "window_1", "window", 0.3, None, shut, 0.65),  # too unsure to call a change
            ("am", "monitor_1", "computer monitor", 0.0, None, '{"screen_on": true}', 0.9),
            ("pm", "monitor_1", "computer monitor", 0.1, None, '{"screen_on": false}', 0.9),  # blurred screens
            ("am", "door_1", "door", 0.0, None, '{"door_open": true}', 0.9),
            ("am", "door_2", "door", 9.0, None, '{"door_open": true}', 0.9),
            ("pm", "door_1", "door", 4.0, None, '{"door_open": false}', 0.9),  # not near either: another door
            ("pm", "door_2", "door", 5.0, None, '{"door_open": false}', 0.9),
        ]
    ).what_changed("am", "pm")
    assert ch["moved"] == []
    assert [c["label"] for c in ch["state_changed"]] == ["gas stove"]


def test_own_name_beats_what_vision_said() -> None:
    mem = _mem()
    mem.con.execute("DELETE FROM objects")
    for oid, label, aliases in [
        ("laptop_1", "laptop", ["keyboard"]),
        ("keyboard_1", "keyboard", []),
        ("bucket_1", "laundry bucket", ["purple bucket"]),
    ]:
        mem.con.execute(
            "INSERT INTO objects (object_id, session_id, label, aliases, last_seen) VALUES (?, 'am', ?, ?, 1.0)",
            (oid, label, json.dumps(aliases)),
        )
    assert [h["label"] for h in mem.search_objects("keyboard")] == ["keyboard"]
    assert mem.search_objects("purple bucket")[0]["seen_as"] == "purple bucket"


def test_places_default_to_the_newest_walk() -> None:
    mem = _mem()
    mem.con.execute("DELETE FROM objects")
    mem.con.execute("INSERT INTO sessions (session_id, scene, captured_at) VALUES ('w1', 'street', '2026-10-28')")
    mem.con.execute("INSERT INTO sessions (session_id, scene, captured_at) VALUES ('w2', 'street', '2026-10-29')")
    for sid, text in [("w1", "Old Shop"), ("w2", "New Shop")]:
        mem.con.execute(
            "INSERT INTO objects (object_id, session_id, label, aliases, text, distance_along_walk_m, side, n_obs) "
            "VALUES ('a', ?, 'signboard', '[]', ?, 5.0, 'left', 1)",
            (sid, text),
        )
    assert [p["name"] for p in mem.places_passed()] == ["New Shop"]
    assert [p["name"] for p in mem.places_passed("w1")] == ["Old Shop"]


def test_places_hide_partial_readings_and_fix_spelling() -> None:
    mem = _mem()
    mem.con.execute("DELETE FROM objects")

    def sign(oid: str, text: str, dist: float, n_obs: int, clear: int = 1, side: str = "left") -> None:
        mem.con.execute(
            "INSERT INTO objects (object_id, session_id, label, aliases, text, text_english, distance_along_walk_m, "
            "side, n_obs, text_clear) VALUES (?, 'walk', 'signboard', '[]', ?, ?, ?, ?, ?, ?)",
            (oid, text, text, dist, side, n_obs, clear),
        )

    sign("a", "Allen Solly", 100.0, 13)
    sign("b", "Alien Solly", 190.0, 10)
    sign("c", "Women I", 120.0, 30, clear=0)
    sign("d", "Vangevar", 250.0, 9)
    sign("e", "Manyavar", 257.0, 34)
    sign("f", "Mangalam", 262.0, 3, side="right")  # other side of the road: its own place
    sign("g", "Lenovo", 278.0, 2)
    assert [p["name"] for p in mem.places_passed("walk")] == [
        "Allen Solly",
        "Allen Solly",
        "Manyavar",
        "Mangalam",
        "Lenovo",
    ]
    assert "Women I" not in [p["name"] for p in mem.route("walk:a", "walk:g")["you_pass"]]


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


def test_route_from_the_start_of_the_walk() -> None:
    mem = _mem()
    mem.con.execute("UPDATE sessions SET start_name='my office (Novel MSR Park)' WHERE session_id='walk'")
    r = mem.route("start", "walk:fruit_stall_1")
    assert (
        r["from"] == "your office (Novel MSR Park)" and r["walk_m"] == 20.0 and r["destination_side"] == "on your right"
    )
    assert any(c["start_name"] for c in mem.list_captures())


def test_search_gives_every_capture_a_turn() -> None:
    mem = _mem()
    for i in range(6):
        mem.con.execute(
            "INSERT INTO objects (object_id, session_id, label, aliases, last_seen) VALUES (?, 'pm', ?, '[]', 99.0)",
            (f"stall_{i}", f"fruit stall {'abcdef'[i]}"),
        )
    sessions = [h["session"] for h in mem.search_objects("fruit stall", limit=2)]
    assert sorted(sessions) == ["pm", "walk"]


def test_places_merge_repeated_readings_and_cap_long_walks() -> None:
    mem = _mem()

    def sign(oid, text, dist, n_obs=1):
        mem.con.execute(
            "INSERT INTO objects (object_id, session_id, label, aliases, text, text_english, distance_along_walk_m, "
            "n_obs, last_seen) VALUES (?, 'walk', 'signboard', '[]', ?, ?, ?, ?, 1.0)",
            (oid, text, text, dist, n_obs),
        )

    sign("a", "Kalyan Jewellers", 100.0, 2)
    sign("b", "KALYAN JEWELLERS", 110.0, 3)
    sign("c", "Kalyan Jewellers", 300.0)  # a second branch further on stays separate
    names = [p["name"] for p in mem.places_passed("walk") if (p["name"] or "").lower().startswith("kalyan")]
    assert len(names) == 2
    capped = mem.places_passed("walk", per_walk=2)
    assert len(capped) == 2 and "walk:b" in [p["id"] for p in capped]  # the most-seen place stays, as read most
    dist = [p["distance_along_walk_m"] for p in capped]
    assert dist == sorted(dist)


def test_my_bag_finds_a_backpack_before_plastic_bags() -> None:
    mem = _mem()
    mem.con.execute(
        "INSERT INTO objects (object_id, session_id, label, aliases, cx, cy, cz, last_seen, confidence, best_obs) "
        "VALUES ('plastic_bag_1', 'am', 'plastic bag', '[]', 0, 0, 0, 9, 0.9, 'o1'), "
        "('backpack_1', 'am', 'backpack', '[]', 0, 0, 0, 5, 0.7, 'o2')"
    )
    for q in ("bag", "my bag", "black bag"):
        assert mem.search_objects(q)[0]["id"] == "am:backpack_1", q
    assert mem.search_objects("plastic bag")[0]["id"] == "am:plastic_bag_1"
    # one way only: a purse is not a backpack
    assert "am:backpack_1" not in [o["id"] for o in mem.search_objects("purse")]
