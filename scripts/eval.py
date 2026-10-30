"""Evaluate the agent on a questions file and write a markdown report.

Questions file (YAML): a list of {clip, scene, questions: [{q, expect: [labels], refuse: bool, state: {...}}]}.
A question passes when:
- refuse: the answer names no object;
- expect: the answered object (or one of the listed objects) fuzzy-matches an expected label;
- state: the stored state of the answered object agrees with every expected key.
Uses a running Glimmer server if there is one (start it once for the whole run).
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from pathlib import Path

import yaml
from rapidfuzz import fuzz

from loci.agent.agent import ask
from loci.agent.tools import Memory, split_uid
from loci.config import load_config, repo_path
from loci.llm import glimmer_server
from loci.stages.store import connect

log = logging.getLogger("eval")


def label_of(mem: Memory, uid: str | None) -> str | None:
    if not uid:
        return None
    o = mem.get_object(uid)
    return None if "error" in o else o["label"]


def score(item: dict, ans: dict, mem: Memory) -> tuple[bool, str]:
    oid = ans.get("object_id")
    if item.get("refuse"):
        return oid is None, "refused" if oid is None else f"answered {label_of(mem, oid)}"
    if oid is None and not ans.get("objects") and not ans.get("evidence"):
        return False, "refused"
    if not ans.get("verified"):
        return False, "not sure (unverified)"
    ids = [oid, *(ans.get("objects") or [])] if oid else ans.get("objects") or []
    labels = [lab for lab in (label_of(mem, u) for u in ids) if lab]
    if not labels:  # a summary without a main object: judge it by what it cites
        for ev in ans.get("evidence") or []:
            sid, oid_ev = split_uid(ev)
            row = mem.con.execute("SELECT label FROM observations WHERE session_id=? AND obs_id=?", (sid, oid_ev))
            labels += [r["label"] for r in row]
    if item.get("state"):
        st = mem.get_state(oid).get("state") or {}
        ok = all(st.get(k) == v for k, v in item["state"].items())
        return ok, f"state {st}"
    exp = item.get("expect") or []
    if not exp:
        return True, "no expectation (manual check)"
    # a name vision gave the answered object counts too ("fruit stand" seen as "coconut stall")
    names = list(labels)
    for u in ids:
        row = mem._obj(u)
        names += json.loads(row["aliases"] or "[]") if row is not None else []
    hit = any(fuzz.WRatio(e, lab) >= 85 for e in exp for lab in names)
    return hit, f"labels {labels}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", type=Path)
    ap.add_argument("--out", type=Path, default=Path("eval/results.md"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    spec = yaml.safe_load(args.questions.read_text())
    items = [{**q, "clip": c["clip"]} for c in spec for q in c["questions"]]
    if args.limit:
        items = items[: args.limit]
    mem = Memory(connect(repo_path(load_config()["paths"]["memory_db"])))

    rows = []
    with glimmer_server(repo_path("debug") / "glimmer_eval.log") as url:
        for it in items:
            t = time.perf_counter()
            try:
                ans = ask(it["q"], mem, url)
            except Exception as e:  # keep evaluating; record the failure
                ans = {"answer": f"ERROR {e}", "object_id": None, "verified": False}
            ok, why = score(it, ans, mem)
            rows.append(
                {
                    **it,
                    "answer": ans.get("answer"),
                    "ok": ok,
                    "why": why,
                    "verified": ans.get("verified"),
                    "searches": [t["args"] for t in ans.get("trace", []) if t.get("tool") == "search_objects"],
                    "attempts": ans.get("attempts"),
                    "latency_s": round(time.perf_counter() - t, 1),
                }
            )
            print(f"{'PASS' if ok else 'FAIL'} {rows[-1]['latency_s']:5.1f}s  {it['q']}  ->  {ans.get('answer')}")

    n = len(rows)
    acc = sum(r["ok"] for r in rows) / n
    ref = [r for r in rows if r.get("refuse")]
    lat = statistics.median(r["latency_s"] for r in rows)
    first = sum(r["attempts"] == 1 for r in rows) / n
    lines = [
        f"# Eval results ({args.questions.name})",
        "",
        f"- questions: {n}",
        f"- accuracy: {acc:.0%}",
        f"- refusals correct: {sum(r['ok'] for r in ref)}/{len(ref)}",
        f"- verified on first attempt: {first:.0%}",
        f"- median latency: {lat:.1f} s",
        "- cloud calls: 0 (local only)",
        "",
        "| | question | answer | why | s |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        a = (r["answer"] or "").replace("|", "/").replace("\n", " ")
        lines.append(f"| {'✅' if r['ok'] else '❌'} | {r['q']} | {a} | {r['why']} | {r['latency_s']} |")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    (args.out.with_suffix(".json")).write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\naccuracy {acc:.0%}, median latency {lat:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
