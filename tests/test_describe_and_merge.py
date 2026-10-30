import json
import sqlite3

from loci.agent.agent import clean_answer
from loci.agent.tools import Memory
from loci.agent.verify import verify
from loci.stages.associate import associate
from loci.stages.attributes import description_aliases, merge_same_signs, wants_description
from loci.stages.store import SCHEMA


def _ob(oid, label, c, t, kf, score=0.9):
    return {
        "obs_id": oid,
        "label": label,
        "centroid": list(c),
        "box_min": [x - 0.05 for x in c],
        "box_max": [x + 0.05 for x in c],
        "t": t,
        "keyframe": kf,
        "score": score,
        "quality": score,
        "area_label": "hallway",
    }


def test_neighbouring_views_of_one_thing_merge_only_when_asked() -> None:
    views = [_ob(f"o{i}", "bucket", (i * 0.6, 0, 0), i * 0.25, i) for i in range(4)]
    assert len(associate(views)) == 4
    assert len(associate(views, adjacent_gap_s=0.8, adjacent_dist_m=1.2)) == 1


def test_two_things_in_one_frame_stay_two() -> None:
    views = [_ob("a", "keyboard", (0, 0, 0), 1.0, 5), _ob("b", "keyboard", (0.8, 0, 0), 1.0, 5)]
    assert len(associate(views, adjacent_gap_s=0.8, adjacent_dist_m=1.2)) == 2


def test_far_apart_in_time_stay_two() -> None:
    views = [_ob("a", "bucket", (0, 0, 0), 1.0, 1), _ob("b", "bucket", (0.6, 0, 0), 9.0, 30)]
    assert len(associate(views, adjacent_gap_s=0.8, adjacent_dist_m=1.2)) == 2


def test_sign_ids_do_not_carry_a_brand() -> None:
    out = associate([_ob("a", "biba store sign", (0, 0, 0), 1.0, 1)])
    assert out[0]["object_id"] == "sign_1" and out[0]["label"] == "biba store sign"


def test_description_aliases() -> None:
    assert description_aliases({"name": "Purple Plastic Bucket", "sells": []}) == ["purple plastic bucket"]
    got = description_aliases({"name": "cart", "sells": ["coconut"]})
    assert {"coconut", "coconut stall", "coconut shop"} <= set(got)
    assert description_aliases({"name": "woman selling fruit", "sells": ["person"]}) == []
    assert description_aliases({}) == []


def test_which_objects_get_described() -> None:
    home = {"describe": "all", "describe_skip": ["door"]}
    assert wants_description("laundry bucket", home) and not wants_description("door", home)
    assert not wants_description("shop sign", home)
    street = {"describe": ["stall", "cart"], "describe_skip": []}
    assert wants_description("fruit stall", street) and not wants_description("tree", street)


def _sign(oid, text, d, side, n, conf=0.9, english=None):
    return {
        "object_id": oid,
        "label": "signboard",
        "text": text,
        "text_english": english,
        "obs_ids": [f"{oid}_{i}" for i in range(n)],
        "n_obs": n,
        "first_seen": d,
        "last_seen": d + 5,
        "confidence": 0.8,
        "centroid": [d, 0.0, 0.0],
        "box_min": [d - 1, -1.0, -1.0],
        "box_max": [d + 1, 1.0, 1.0],
        "best_obs": f"{oid}_0",
        "attribute_confidence": conf,
        "attribute_obs": f"{oid}_0",
        "distance_along_walk_m": d,
        "side": side,
        "passed_at": d,
        "location_text": f"on your {side}, about {d:.0f} m into the walk",
    }


def test_same_reading_nearby_is_one_shop() -> None:
    objs = [
        _sign("s1", "BIBA", 173.0, "left", 8),
        _sign("s2", "ಬಿಬಾ", 187.0, "left", 20, conf=0.95, english="BIBA"),
        _sign("s3", "BIBA", 189.0, "right", 2),
        _sign("s4", "Lenovo", 180.0, "left", 5),
    ]
    out = merge_same_signs(objs)
    assert sorted(o["object_id"] for o in out) == ["s2", "s4"]
    biba = next(o for o in out if o["object_id"] == "s2")
    assert biba["n_obs"] == 30 and len(biba["obs_ids"]) == 30
    assert biba["side"] == "left" and biba["distance_along_walk_m"] == 187.0
    assert biba["text"] == "ಬಿಬಾ" and biba["aliases"] == []
    assert biba["location_text"] == "on your left, about 187 m into the walk"


def test_part_of_a_name_is_the_same_shop_but_distance_keeps_branches_apart() -> None:
    out = merge_same_signs([_sign("a", "KALYAN", 131.0, "left", 17), _sign("b", "KALYAN JEWELLERS", 144.0, "left", 28)])
    assert len(out) == 1 and out[0]["n_obs"] == 45
    far = merge_same_signs([_sign("a", "Allen Solly", 114.0, "right", 5), _sign("b", "Allen Solly", 193.0, "right", 4)])
    assert len(far) == 2


def test_words_any_shop_uses_do_not_make_one_shop() -> None:
    objs = [
        _sign("a", "Reebok OUTLET", 300.0, "left", 5),
        _sign("b", "Levi's OUTLET", 305.0, "left", 5),
        _sign("c", "OUTLET", 310.0, "left", 5),
        _sign("d", "FACTORY OUTLETS", 312.0, "left", 5),
        _sign("e", "Levi's", 314.0, "left", 9),
    ]
    out = merge_same_signs(objs)
    assert sorted(o["object_id"] for o in out) == ["a", "c", "d", "e"]
    assert next(o for o in out if o["object_id"] == "e")["n_obs"] == 14


def test_a_named_reading_beats_a_clearer_generic_one() -> None:
    objs = [_sign("a", "OUTLET", 10.0, "left", 9, conf=0.99), _sign("b", "OUTLET", 12.0, "left", 2, conf=0.5)]
    objs += [_sign("c", "BIBA", 12.0, "left", 2, conf=0.5), _sign("d", "BIBA store", 13.0, "left", 5, conf=0.9)]
    out = merge_same_signs(objs)
    assert {o["text"] for o in out} == {"OUTLET", "BIBA store"}


def test_unread_signs_are_left_alone() -> None:
    out = merge_same_signs([_sign("a", None, 10.0, "left", 3), _sign("b", None, 11.0, "left", 3)])
    assert len(out) == 2


def _insert(con, oid, sid, label, aliases, c=(0, 0, 0), last_seen=1.0):
    con.execute(
        "INSERT INTO objects (object_id, session_id, label, aliases, area, cx, cy, cz, box_min, box_max, last_seen,"
        " best_obs) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (oid, sid, label, json.dumps(aliases), "street", *c, "[0,0,0]", "[1,1,1]", last_seen, f"obs_{oid}"),
    )


def _mem() -> Memory:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.execute("INSERT INTO sessions (session_id, scene, start_name) VALUES ('s', 'street', 'my office')")
    return Memory(con)


def test_what_vision_saw_outranks_a_synonym() -> None:
    mem = _mem()
    for i in range(4):
        _insert(mem.con, f"stall_{i}", "s", "street stall", [], last_seen=50.0 + i)
    _insert(mem.con, "fruit_stand_1", "s", "fruit stand", ["coconut", "coconut stall"], last_seen=1.0)
    top = mem.search_objects("coconut stall")[0]
    assert top["id"] == "s:fruit_stand_1" and top["seen_as"] == "coconut stall"


def test_colour_finds_the_described_object() -> None:
    mem = _mem()
    _insert(mem.con, "bucket_1", "s", "laundry bucket", ["purple plastic bucket"])
    _insert(mem.con, "bucket_2", "s", "laundry bucket", ["blue plastic bucket"], last_seen=9.0)
    found = mem.search_objects("purple bucket")
    assert found[0]["id"] == "s:bucket_1" and found[0]["seen_as"] == "purple plastic bucket"


def test_desk_with_things_on_it_comes_first() -> None:
    mem = _mem()
    _insert(mem.con, "desk_1", "s", "desk", [], last_seen=1.0)
    _insert(mem.con, "desk_2", "s", "desk", [], last_seen=9.0)
    mem.con.execute("INSERT INTO relations VALUES ('s', 'laptop_1', 'on', 'desk_1', 0.1)")
    found = mem.search_objects("desk")
    assert found[0]["id"] == "s:desk_1" and found[0]["things_on"] == 1
    assert "things_on" not in found[1]


def test_captures_speak_to_the_user() -> None:
    assert _mem().list_captures()[0]["start_name"] == "your office"


def test_clean_answer() -> None:
    assert clean_answer("Seen at 2026-10-28T16:45:46+05:30 in the hall.") == "Seen at Oct 28 in the hall."
    assert clean_answer("In the 2026-10-29 walk.") == "In the Oct 29 walk."
    assert clean_answer("About 173 m in at 2:49; street2:shop_sign_91") == "About 173 m in at 2:49"
    assert clean_answer("The stove (home2:gas_stove_1) had burner_lit false at 0:18.") == (
        "The stove had burner lit false at 0:18."
    )
    assert clean_answer("ಸ್ಟವ್ 0:18 ರಂತೆ") == "ಸ್ಟವ್ 0:18 ರಂತೆ"


def test_on_your_left_is_not_the_state_on() -> None:
    mem = _mem()
    _insert(mem.con, "stall_1", "s", "fruit stand", [])
    mem.con.execute("""UPDATE objects SET state='{"shop_open": true}', state_seen_at=2.0 WHERE object_id='stall_1'""")
    mem.con.execute(
        "INSERT INTO observations (obs_id, session_id, object_id, keyframe, t, label, score)"
        " VALUES ('obs_stall_1','s','stall_1',0,2.0,'fruit stand',0.9)"
    )
    ans = {
        "answer": "No, the stall on your left was closed at 0:02.",
        "object_id": "s:stall_1",
        "evidence": ["obs_stall_1"],
    }
    assert any("stored state" in p for p in verify(ans, mem))


def test_a_burner_side_is_not_a_relation() -> None:
    mem = _mem()
    _insert(mem.con, "stove_1", "s", "gas stove", [])
    mem.con.execute(
        "INSERT INTO observations (obs_id, session_id, object_id, keyframe, t, label, score)"
        " VALUES ('obs_stove_1','s','stove_1',0,1.0,'gas stove',0.9)"
    )
    ans = {"answer": "It went from on the right burner to off.", "object_id": "s:stove_1", "evidence": ["obs_stove_1"]}
    assert not [p for p in verify(ans, mem) if "relation" in p]
