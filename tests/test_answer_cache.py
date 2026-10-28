import sqlite3

from loci import answer_cache
from loci.stages.store import SCHEMA


def test_cache_roundtrip_and_invalidation(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(answer_cache, "_path", lambda: tmp_path / "cache.json")
    con = sqlite3.connect(":memory:")
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    con.execute("INSERT INTO sessions (session_id, scene, captured_at) VALUES ('home', 'home', '2026-10-28T09:00')")
    answer_cache.put("Is the stove on?", con, {"answer": "As of 0:15 it was lit.", "verified": True})
    answer_cache.put("Where are my keys?", con, {"answer": "?", "verified": False})  # unverified: not cached
    assert answer_cache.get("is the stove on", con)["cached"] is True
    assert answer_cache.get("Where are my keys?", con) is None
    con.execute("INSERT INTO sessions (session_id, scene, captured_at) VALUES ('home2', 'home', '2026-10-28T18:00')")
    assert answer_cache.get("Is the stove on?", con) is None  # new capture: memory changed
