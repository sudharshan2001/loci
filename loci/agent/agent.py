"""The Loci agent: Glimmer with tools over the memory, then a plain-code verifier.

Flow: tool-calling loop -> final JSON -> code fills position/robot goal from the memory -> verify ->
one retry with the problems -> otherwise a "not sure" answer with the last confirmed sighting.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from loci.agent.tools import Memory, fmt_time, split_uid
from loci.agent.verify import verify
from loci.llm import chat, extract_json

log = logging.getLogger(__name__)

SYSTEM = """You are Loci, a private spatial memory for AI glasses and robots. You answer questions about \
where things are and what state they were in, using ONLY the memory tools. Captures are walks through a home \
or along a street; times like 0:31 are seconds into that capture.

Rules:
- Always call search_objects first. Then get_object / get_state / places_passed / objects_on / search_notes \
as needed. Never invent objects, positions, names or times.
- Cite evidence: observation ids returned by the tools (e.g. obs_0012_03) for the object you answer about.
- States (stove, light, tap, door, screen) are only known as of when they were seen. Say "when you left \
the kitchen at 0:31, ..." or "as of 0:31". Never say what the state is now.
- A place can change between captures. For state questions call state_history and answer from the newest \
capture, giving its time; if an older capture differs, mention it ("it was off in the morning capture").
- "What changed / moved since ...": call list_captures, then what_changed with the earlier and the later \
capture of that place.
- Directions between two places on a walk: call route and give distance, direction and one or two landmarks.
- Street answers: give the shop name as read from the sign, the side (on your left/right), the distance into \
the walk in metres, and the time.
- Use the location_text words from the tools for where things are (on the desk, left of the monitor ...).
- If nothing matching was seen, say so plainly: "I haven't seen a <thing> in any capture." and set \
object_id to null.
- Keep the answer to one or two short spoken-style sentences.

When you are done, reply with ONLY this JSON:
{"answer": str, "object_id": "<session>:<object>" or null, "objects": [other ids mentioned], \
"location_text": str or null, "seen_at": "m:ss" or null, "evidence": [observation ids], "confidence": 0..1}"""

TOOLS = [
    (
        "search_objects",
        "Find objects by name or sign text across all captures (fuzzy, synonyms).",
        {"query": {"type": "string"}},
        ["query"],
    ),
    (
        "get_object",
        "Position, relations, location text, first/last seen and evidence for one object.",
        {"id": {"type": "string"}},
        ["id"],
    ),
    (
        "get_state",
        "Stored state of a stateful object (stove, light, door, tap, screen) and when it was seen.",
        {"id": {"type": "string"}},
        ["id"],
    ),
    (
        "places_passed",
        "Shop signs passed on a street walk, in order, with distance, side and time.",
        {"session_id": {"type": "string"}},
        [],
    ),
    (
        "objects_on",
        "Objects on or inside an anchor object (desk, table, shelf, drawer...).",
        {"anchor_id": {"type": "string"}},
        ["anchor_id"],
    ),
    (
        "objects_near",
        "Objects within radius_m metres of an object, nearest first.",
        {"id": {"type": "string"}, "radius_m": {"type": "number"}},
        ["id"],
    ),
    (
        "history",
        "Every sighting of objects matching a label across captures.",
        {"label": {"type": "string"}},
        ["label"],
    ),
    (
        "state_history",
        "States of matching objects (stove, lights, laundry, shop open) in every capture, newest first.",
        {"query": {"type": "string"}},
        ["query"],
    ),
    (
        "route",
        "Walking directions between two things seen on the same street walk: distance, direction, what you pass.",
        {"from_id": {"type": "string"}, "to_id": {"type": "string"}},
        ["from_id", "to_id"],
    ),
    (
        "what_changed",
        "What changed between two captures of the same place: added, removed, moved, state changes.",
        {"session_a": {"type": "string", "description": "earlier capture"}, "session_b": {"type": "string"}},
        ["session_a", "session_b"],
    ),
    ("list_captures", "All captures with their scene type and capture time, oldest first.", {}, []),
    (
        "search_notes",
        "Things the user said out loud during captures, with the objects in view.",
        {"text": {"type": "string"}},
        ["text"],
    ),
]


def tool_schema() -> list[dict]:
    return [
        {
            "type": "function",
            "function": {
                "name": n,
                "description": d,
                "parameters": {"type": "object", "properties": p, "required": r},
            },
        }
        for n, d, p, r in TOOLS
    ]


def call_tool(mem: Memory, name: str, args: dict[str, Any]) -> Any:
    fn = getattr(mem, name, None)
    if fn is None or name not in {t[0] for t in TOOLS}:
        return {"error": f"unknown tool {name}"}
    try:
        result = fn(**args)
    except TypeError as e:
        return {"error": f"bad arguments for {name}: {e}"}
    if result == []:
        return {
            "results": [],
            "note": "Nothing matching in any capture. Try at most one other wording; "
            "if that also finds nothing, answer that it was not seen.",
        }
    return result


def complete(ans: dict[str, Any], mem: Memory) -> dict[str, Any]:
    """Fill position, robot goal and scene facts from the memory (the model never writes numbers here)."""
    oid = ans.get("object_id")
    ans.setdefault("used_cloud", False)
    if not oid:
        ans.update(position_m=None, robot_goal=None)
        return ans
    o = mem.get_object(oid)
    if "error" in o:
        return ans
    sid, _ = split_uid(oid)
    ans["position_m"] = o["position_m"]
    ans["location_text"] = ans.get("location_text") or o["location_text"]
    ans["session"] = sid
    ans["robot_goal"] = {
        "frame": f"loci/{sid}",
        "target": o["label"],
        "position_m": o["position_m"],
        "extent_m": o["extent_m"],
        "units": "metres",
        "axes": "x right, y down, z forward of the capture's first camera",
    }
    return ans


def not_sure(mem: Memory, question: str, candidates: list[dict], problems: list[str]) -> dict[str, Any]:
    if candidates:
        c = candidates[0]
        o = mem.get_object(c["id"])
        text = (
            f"I'm not sure. The last confirmed sighting of the {o['label']} was "
            f"{o['location_text']}, at {o['last_seen']}."
        )
        return complete(
            {
                "answer": text,
                "object_id": c["id"],
                "evidence": o["evidence"][:1],
                "seen_at": o["last_seen"],
                "confidence": 0.3,
                "verified": False,
                "verifier_problems": problems,
            },
            mem,
        )
    return complete(
        {
            "answer": "I'm not sure, and I couldn't find it in any capture.",
            "object_id": None,
            "evidence": [],
            "confidence": 0.1,
            "verified": False,
            "verifier_problems": problems,
        },
        mem,
    )


def ask(question: str, mem: Memory, base_url: str, max_steps: int = 8, strength: str = "medium") -> dict[str, Any]:
    t0 = time.perf_counter()
    trace: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    candidates: list[dict] = []
    for attempt in range(2):
        ans = None
        for step in range(max_steps):
            # last step: no tools offered, so the model has to give its final answer
            last = step == max_steps - 1
            if last:
                messages.append({"role": "user", "content": "Stop using tools. Reply with only the final JSON now."})
            msg = chat(base_url, messages, strength=strength, **({} if last else {"tools": tool_schema()}))
            calls = msg.get("tool_calls") or []
            messages.append(
                {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
                if calls
                else {"role": "assistant", "content": msg.get("content") or ""}
            )
            if not calls:
                try:
                    ans = extract_json(msg.get("content") or "")
                except ValueError:
                    messages.append({"role": "user", "content": "Reply with only the final JSON object."})
                    continue
                break
            for c in calls:
                name = c["function"]["name"]
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = call_tool(mem, name, args)
                if name == "search_objects" and isinstance(result, list):
                    candidates = candidates or result
                trace.append({"tool": name, "args": args, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": c.get("id", name),
                        "content": json.dumps(result, ensure_ascii=False)[:6000],
                    }
                )
        if not isinstance(ans, dict):
            problems = ["agent did not produce a final answer"]
        else:
            ans = complete(ans, mem)
            problems = verify(ans, mem)
            if not problems:
                ans.update(
                    verified=True,
                    trace=trace,
                    attempts=attempt + 1,
                    latency_s=round(time.perf_counter() - t0, 1),
                )
                return ans
        log.info("verifier rejected attempt %d: %s", attempt + 1, problems)
        trace.append({"verifier": problems})
        messages.append(
            {
                "role": "user",
                "content": "The verifier found problems with that answer:\n- "
                + "\n- ".join(problems)
                + "\nFix them using the tools and reply with the JSON again.",
            }
        )
    out = not_sure(mem, question, candidates, problems)
    out.update(trace=trace, attempts=2, latency_s=round(time.perf_counter() - t0, 1))
    return out


__all__ = ["ask", "fmt_time"]
