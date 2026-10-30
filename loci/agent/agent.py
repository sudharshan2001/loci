"""The Loci agent: Glimmer with tools over the memory, then a plain-code verifier.

Flow: tool-calling loop -> final JSON -> code fills position/robot goal from the memory -> verify ->
one retry with the problems -> otherwise a "not sure" answer with the last confirmed sighting.
"""

from __future__ import annotations

import json
import logging
import re
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
- Cite evidence: the evidence ids returned by the tools (e.g. home:obs_0012_03), copied exactly, from the same \
capture as the object you answer about.
- States (stove, light, tap, door, screen) are only known as of when they were seen. Say "when you left \
the kitchen at 0:31, ..." or "as of 0:31". Never say what the state is now.
- "Where did I leave X?" / "Is X still there?": call last_known. Answer from the newest sighting with its \
capture time; if it is movable and the sighting is old, say so ("that was this morning; keys move, so check").
- A place can change between captures. For state questions call state_history and answer from the newest \
capture, giving its time; if an older capture differs, mention it ("it was off in the morning capture").
- "What changed / moved since ...": call list_captures, then what_changed with the earlier and the later \
capture of that place. Answer from its result without more tools: state changes first (the stove was lit, now \
off), then what moved, then a few additions. Put the ids you mention in objects.
- "Into the walk" distances are measured from where the walk started; list_captures gives each walk's \
start_name (e.g. the user's office). For "how far from my office/start", call route with from_id "start".
- Directions between two places on a walk: call route and give distance, direction and one or two landmarks.
- Street answers: give the shop name as read from the sign, the side (on your left/right), the distance into \
the walk in metres, and the time.
- Use the location_text words from the tools for where things are (on the desk, left of the monitor ...).
- A search result with seen_as was seen to be that thing (a "fruit stand" seen as "coconut stall"): answer about it \
without look. Several results seen_as the same thing: use the one seen most recently.
- "What's on my desk/table?": take the desk with the most things_on across all captures, call objects_on on it, \
name each kind of thing once and say which room. Do not answer from a desk with no things_on when another has some.
- If search returns things of a related kind but not the exact name (a "street stall" or "fruit stand" when \
asked about a coconut stall), call look on the likely candidates before deciding. Answer about the one that \
look confirms; only say it was not seen if look rules them all out.
- "What shops did I pass?" with no walk named: call list_captures, then places_passed for the newest walk only. \
Name a handful in walk order with side and distance, say which walk it was, and put their ids in objects.
- Kinds of shop ("clothing stores", "jewellers"): call places_passed and choose from the names by what you know \
of them (Allen Solly and Louis Philippe sell clothes); give each one's name, side and distance.
- If nothing matching was seen, say so plainly: "I haven't seen a <thing> in any capture." and set \
object_id to null.
- Write for a person: dates like "Oct 28", no ids, no field names (say "the burner was not lit", not burner_lit), \
no capture names (say "your Oct 29 walk", not street2), and nothing about how it is stored ("logged as", "seen as").
- Keep the answer to one or two short spoken-style sentences, in the same language as the question (answer a \
Kannada, Tamil or Hindi question in that language; keep shop names as written on the sign).

When you are done, reply with ONLY this JSON:
{"answer": str, "object_id": "<session>:<object>" or null, "objects": [other ids mentioned], \
"location_text": str or null, "seen_at": "m:ss" or null, "evidence": [observation ids], "confidence": 0..1}"""

SCRIPTS = {
    "Hindi": (0x0900, 0x097F),
    "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F),
    "Kannada": (0x0C80, 0x0CFF),
}


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
ISO_DATE = re.compile(r"\b\d{4}-(\d{2})-(\d{2})(?:T\d{2}:\d{2}(?::\d{2})?(?:[+-]\d{2}:\d{2}|Z)?)?")
OBJECT_ID = re.compile(r"[;,]?\s*\(?\b[a-z0-9_]+:[a-z0-9]+(?:_[a-z0-9]+)+\b\)?")
FIELD_NAME = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")


def clean_answer(text: str) -> str:
    """What is said to a person: no raw timestamps, ids or field names, even when the model copies them from a tool."""
    text = ISO_DATE.sub(lambda m: f"{MONTHS[int(m[1]) - 1]} {int(m[2])}", text)
    text = OBJECT_ID.sub("", text)
    return FIELD_NAME.sub(lambda m: m[0].replace("_", " "), text).strip()


def script_counts(text: str) -> dict[str, int]:
    return {lang: sum(lo <= ord(c) <= hi for c in text) for lang, (lo, hi) in SCRIPTS.items()}


def question_language(question: str) -> str | None:
    counts = script_counts(question)
    lang = max(counts, key=lambda k: counts[k])
    return lang if counts[lang] else None


def language_problems(question: str, answer: str) -> list[str]:
    """An Indian-language question gets its answer mostly in that language (shop names may stay as written)."""
    lang = question_language(question)
    if lang is None:
        return []
    counts = script_counts(answer)
    if counts[lang] and counts[lang] >= max(counts.values()):
        return []
    return [f"the question is in {lang}: write the answer in {lang}, keeping only shop names as written"]


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
        'Walking directions between two things seen on the same street walk (from_id may be "start", the start '
        "of the walk): distance, direction, what you pass.",
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
        "look",
        "Look at an object's redacted evidence photo and answer a visual question about it (e.g. is this a "
        "coconut stall? is the shutter up?).",
        {"id": {"type": "string"}, "question": {"type": "string"}},
        ["id", "question"],
    ),
    (
        "last_known",
        "The same thing across all captures: newest sighting, earlier sightings, whether it moved, and how likely "
        "it is still there (movable things like keys fade within hours; furniture does not).",
        {"query": {"type": "string"}},
        ["query"],
    ),
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


def look(mem: Memory, base_url: str, id: str, question: str) -> dict[str, Any]:
    """Glimmer looks at the object's best redacted observation (tight, context and wide crops)."""
    from loci.providers import get_vision_checker
    from loci.session import Session
    from loci.stages.attributes import save_crops

    row = mem._obj(id)
    if row is None:
        return {"error": f"no object {id}"}
    s = Session.named(row["session_id"])
    ob = next((o for o in s.read(s.observations) if o["obs_id"] == row["best_obs"]), None)
    if ob is None:
        return {"error": f"no observation for {id}"}
    c = save_crops(s, ob)
    ans = get_vision_checker(base_url).check(
        c["ctx"],
        f'{question} Return JSON {{"answer": short answer, "yes": true/false}}.',
        tight=c["tight"],
        wide=c["wide"],
    )
    return {
        "id": id,
        "answer": ans.answer,
        "confidence": round(ans.confidence, 2),
        "evidence": [row["best_obs"]],
        "checked_by": "+".join(ans.attempts),
    }


def qualify_evidence(result: Any, sid: str | None = None) -> Any:
    """Prefix evidence ids with their capture ("<session>:obs_...") everywhere in a tool result.

    Observation ids are only unique within a capture; without the prefix the agent can pair one capture's
    evidence with another capture's object.
    """
    if isinstance(result, list):
        return [qualify_evidence(x, sid) for x in result]
    if not isinstance(result, dict):
        return result
    own = result.get("session") or result.get("capture")
    if not own and isinstance(result.get("id"), str) and ":" in result["id"]:
        own = result["id"].rpartition(":")[0]
    own = own or sid
    out = {}
    for k, v in result.items():
        if k == "evidence" and isinstance(v, list) and own:
            out[k] = [e if ":" in e else f"{own}:{e}" for e in v]
        else:
            out[k] = qualify_evidence(v, own)
    return out


def call_tool(mem: Memory, name: str, args: dict[str, Any], base_url: str | None = None) -> Any:
    if name == "look":
        if base_url is None:
            return {"error": "look needs the vision model"}
        try:
            return qualify_evidence(look(mem, base_url, **args))
        except TypeError as e:
            return {"error": f"bad arguments for look: {e}"}
    fn = getattr(mem, name, None)
    if fn is None or name not in {t[0] for t in TOOLS} - {"look"}:
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
    return qualify_evidence(result)


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
        "axes": "x right, y down, z forward of the capture's first camera, gravity-aligned",
    }
    if o.get("scene") == "home":
        try:  # a path for a ground robot; never let planning break an answer
            from loci.nav_run import navigate
            from loci.session import Session

            nav = navigate(Session.named(sid), oid.split(":", 1)[1])
            if nav.get("ok"):
                ans["robot_goal"].update(
                    approach_point_m=nav["approach_point_m"],
                    waypoints_m=nav["waypoints_m"],
                    path_length_m=nav["length_m"],
                    through_unscanned_fraction=nav["through_unscanned_fraction"],
                )
        except Exception as e:  # noqa: BLE001
            log.warning("navigation failed for %s: %s", oid, e)
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


def ask(
    question: str, mem: Memory, base_url: str, max_steps: int = 8, strengths: tuple[str, str] = ("low", "medium")
) -> dict[str, Any]:
    """Answer with tools. The first attempt reasons at low strength (fast); a verifier retry escalates."""
    t0 = time.perf_counter()
    trace: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    lang = question_language(question)
    if lang:
        messages[-1]["content"] = f"{question}\n\n(Answer in {lang}.)"
    candidates: list[dict] = []
    for attempt in range(2):
        ans = None
        for step in range(max_steps):
            # last step: no tools offered, so the model has to give its final answer
            last = step == max_steps - 1
            if last:
                messages.append({"role": "user", "content": "Stop using tools. Reply with only the final JSON now."})
            msg = chat(base_url, messages, strength=strengths[attempt], **({} if last else {"tools": tool_schema()}))
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
                result = call_tool(mem, name, args, base_url)
                if name == "search_objects" and isinstance(result, list):
                    candidates = candidates or result
                trace.append({"tool": name, "args": args, "result": result})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": c.get("id", name),
                        "content": json.dumps(result, ensure_ascii=False)[:9000],
                    }
                )
        if not isinstance(ans, dict):
            problems = ["agent did not produce a final answer"]
        else:
            ans = complete(ans, mem)
            ans["answer"] = clean_answer(str(ans.get("answer") or ""))
            problems = verify(ans, mem) + language_problems(question, str(ans.get("answer") or ""))
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
