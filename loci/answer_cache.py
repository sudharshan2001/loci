"""Answer cache for live demos: same question + same memory = same verified answer, instantly.

The key includes a fingerprint of the memory, so rebuilding a session or forgetting an object invalidates it.
Only verified answers are cached.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from loci.config import repo_path


def fingerprint(con: sqlite3.Connection) -> str:
    rows = con.execute(
        "SELECT session_id, captured_at, created_at, (SELECT COUNT(*) FROM objects o WHERE o.session_id=s.session_id)"
        " FROM sessions s ORDER BY session_id"
    ).fetchall()
    return hashlib.sha1(json.dumps([tuple(r) for r in rows]).encode()).hexdigest()[:12]


def _key(question: str, con: sqlite3.Connection) -> str:
    q = re.sub(r"[^\w\s]", "", question.lower()).split()
    return f"{fingerprint(con)}:{' '.join(q)}"


def _path() -> Path:
    return repo_path("data") / "answer_cache.json"


def get(question: str, con: sqlite3.Connection) -> dict[str, Any] | None:
    p = _path()
    if not p.exists():
        return None
    hit = json.loads(p.read_text()).get(_key(question, con))
    return {**hit, "cached": True} if hit else None


def put(question: str, con: sqlite3.Connection, ans: dict[str, Any]) -> None:
    if not ans.get("verified"):
        return
    p = _path()
    data = json.loads(p.read_text()) if p.exists() else {}
    data[_key(question, con)] = ans
    p.write_text(json.dumps(data, ensure_ascii=False))
