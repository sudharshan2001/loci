import json
import sqlite3

from loci.agent.agent import clean_answer, own_evidence_first
from loci.agent.leaving import facts, is_personal, worth_flagging
from loci.agent.tools import Memory
from loci.stages.associate import settled_area
from loci.stages.store import SCHEMA, text_is_clear


def test_clean_answer_wording() -> None:
    assert clean_answer("I haven't seen a car keys in any capture.") == "I haven't seen any car keys in any capture."
    assert clean_answer("I haven't seen a passport in any capture.") == "I haven't seen a passport in any capture."
    assert clean_answer("I haven't seen a glass in any capture.") == "I haven't seen a glass in any capture."
    assert clean_answer("On your street2 walk, home2.") == "On your street walk, home."
    assert clean_answer("The fruit stand was seen as a coconut stall.") == "The fruit stand was a coconut stall."
    assert clean_answer("The stall – seen as the fruit stand – was open.") == "The stall – the fruit stand – was open."


def test_proof_frame_comes_from_the_answered_capture() -> None:
    assert own_evidence_first(["home:obs_1", "home2:obs_2", "obs_3"], "home2") == ["home2:obs_2", "obs_3", "home:obs_1"]


def test_settled_area() -> None:
    assert settled_area("gas stove", ["office", "office"]) == "kitchen"
    assert settled_area("desk", ["kitchen", "kitchen"]) == "room"
    assert settled_area("laptop", ["office", "office", "bedroom"]) == "office"
    assert settled_area("chair", ["office", "bedroom"]) == "room"  # no majority
    assert settled_area("spice jar", ["kitchen", "kitchen"]) == "kitchen"


def test_text_is_clear() -> None:
    answers = {"o1|sign": {"answer": {"readable": False}}, "o2|sign": {"answer": {"readable": True}}}
    assert text_is_clear({"text": None}, answers) is None
    assert text_is_clear({"text": "Women I", "attribute_obs": "o1"}, answers) == 0
    assert text_is_clear({"text": "Lenovo", "attribute_obs": "o2"}, answers) == 1
    assert text_is_clear({"text": "H Javed ... Hair", "attribute_obs": "o2"}, answers) == 0
    assert text_is_clear({"text": "Mr. [illegible]", "attribute_obs": "o2"}, answers) == 0


def test_what_is_worth_flagging() -> None:
    assert worth_flagging({"burner_lit": True}, 0.75)
    assert not worth_flagging({"burner_lit": False}, 0.9)
    assert not worth_flagging({"door_open": True}, 0.9)
    assert not worth_flagging({"screen_on": True}, 0.9)
    assert not worth_flagging({"window_open": True}, 0.75)
    assert worth_flagging({"window_open": True}, 0.9)
    assert is_personal("keys") and is_personal("backpack")
    assert not is_personal("plastic bag") and not is_personal("laptop")


def test_facts_keep_one_per_kind() -> None:
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.execute("INSERT INTO sessions (session_id, scene) VALUES ('home', 'home')")
    rows = [
        ("window_1", "window", '{"window_open": true}', 0.9, "kitchen"),
        ("window_2", "window", '{"window_open": true}', 0.85, "kitchen"),
        ("door_1", "door", '{"door_open": true}', 0.9, "kitchen"),
        ("backpack_1", "backpack", None, None, "room"),
        ("backpack_2", "backpack", None, None, "room"),
        ("plastic_bag_1", "plastic bag", None, None, "room"),
    ]
    for oid, label, state, conf, area in rows:
        con.execute(
            "INSERT INTO objects (object_id, session_id, label, aliases, area, state, state_seen_at,"
            " attribute_confidence, best_obs, n_obs, last_seen) VALUES (?, 'home', ?, '[]', ?, ?, 1.0, ?, ?, 2, 3.0)",
            (oid, label, area, state, conf, f"obs_{oid}"),
        )
    got = facts(Memory(con), "home")
    assert sorted(f["label"] for f in got) == ["backpack", "window"]
    assert json.dumps(got)  # goes to the model as JSON
