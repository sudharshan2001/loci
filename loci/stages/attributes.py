"""Stage 9: Glimmer reads signs and checks the state of stateful objects, from redacted crops.

Crops (tight / context / wide) go to <session>/crops/ and are reused as evidence. A state is stored with the
timestamp of the observation it came from: answers say "as of <time>", never "now".
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from loci.config import load_config
from loci.providers.base import VisionChecker
from loci.session import Session

log = logging.getLogger(__name__)

SIGN_LABELS = {"shop sign", "signboard", "sign", "store sign", "shop name board"}

SIGN_Q = (
    "This is a crop of a shop sign. Read the shop's name exactly as written (it may be English, Kannada or "
    'both). Return JSON {"text": name as written, "text_english": English transliteration or translation, '
    '"readable": true/false}.'
)

# keyword in the label -> question and the state keys it should return
STATE_QS: list[tuple[tuple[str, ...], str]] = [
    (
        ("laundry", "clothes pile", "pile of clothes", "washing machine"),
        "Is there still laundry here waiting (clothes piled up or in the bucket/machine)? Return JSON "
        '{"laundry_pending": true/false}.',
    ),
    (
        ("shop front", "stall", "stand", "cart", "shop entrance", "storefront", "kiosk"),
        "Is this shop or stall open for business (shutter up, goods on display, lights on)? Return JSON "
        '{"shop_open": true/false}.',
    ),
    (
        ("stove", "burner", "flame", "hob", "cooktop"),
        "Is any burner lit (visible flame)? Is anything (pan, pot) on a lit burner? Return JSON "
        '{"burner_lit": true/false, "anything_on_burner": true/false, "which_burner": "left/right/front/back/'
        'unknown"}.',
    ),
    (
        ("lamp", "light", "bulb", "tube light"),
        'Is this light switched on? Return JSON {"light_on": true/false}.',
    ),
    (("tap", "faucet"), 'Is water running from this tap? Return JSON {"water_running": true/false}.'),
    (("door", "gate"), 'Is this door or gate open or closed? Return JSON {"door_open": true/false}.'),
    (("window",), 'Is this window open or closed? Return JSON {"window_open": true/false}.'),
    (
        ("tv", "television", "monitor", "screen"),
        "Is this screen switched on (lit, showing anything)? The content may be blurred for privacy; "
        'judge only whether it is lit. Return JSON {"screen_on": true/false}.',
    ),
    (("fan",), 'Is this fan running (blurred blades)? Return JSON {"fan_on": true/false}.'),
]


def question_for(label: str, scene: str = "home") -> tuple[str, str] | None:
    """('sign' | 'state', question) for labels that need a visual attribute check.

    Open/closed applies to shops and stalls on the street only (a "monitor stand" at home is not a shop).
    """
    if label in SIGN_LABELS or label.endswith(" sign") or label.endswith("signboard"):
        return "sign", SIGN_Q
    for keys, q in STATE_QS:
        if "shop_open" in q and scene != "street":
            continue
        if any(k in label for k in keys):
            return "state", q
    return None


def crop(img, bbox: list[int], pad: float) -> Any:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    px, py = int(bw * pad), int(bh * pad)
    return img[max(0, y0 - py) : min(h, y1 + py), max(0, x0 - px) : min(w, x1 + px)]


def save_crops(session: Session, ob: dict) -> dict[str, Path]:
    out = session.root / "crops"
    out.mkdir(exist_ok=True)
    img = cv2.imread(str(session.keyframes_dir / f"{ob['keyframe']:04d}.jpg"))
    paths = {}
    for name, pad in (("tight", 0.05), ("ctx", 0.4), ("wide", 1.5)):
        p = out / f"{ob['obs_id']}_{name}.jpg"
        if not p.exists():
            cv2.imwrite(str(p), crop(img, ob["bbox"], pad), [cv2.IMWRITE_JPEG_QUALITY, 92])
        paths[name] = p
    return paths


def settings(scene: str) -> dict[str, Any]:
    """Per-scene attribute budget from config (attributes.<scene>)."""
    cfg = load_config().get("attributes", {}).get(scene, {})
    return {
        "per_object": cfg.get("per_object", 2),
        "skip": set(cfg.get("skip", [])),
        "second_budget_s": cfg.get("second_budget_s"),
        "min_confidence": cfg.get("min_confidence", 0.7),
    }


def targets(session: Session) -> list[tuple[dict, str, str, list[dict]]]:
    """(object, kind, question, observations to ask about) for every object that needs a vision check."""
    scene = session.scene()
    cfg = settings(scene)
    obs = {o["obs_id"]: o for o in session.read(session.observations)}
    out = []
    for o in session.read(session.objects):
        q = question_for(o["label"], scene)
        if q is None or o["label"] in cfg["skip"]:
            continue
        cands = sorted((obs[i] for i in o["obs_ids"]), key=lambda m: -m["quality"])[: cfg["per_object"]]
        out.append((o, q[0], q[1], cands))
    return out


def pending(session: Session) -> int:
    """How many vision checks are not yet cached (0 means no Glimmer needed)."""
    cache_path = session.root / "attr_cache.json"
    cache = session.read(cache_path) if cache_path.exists() else {}
    return sum(f"{c['obs_id']}|{kind}" not in cache for _, kind, _, cands in targets(session) for c in cands)


def second_priority(o: dict, ob: dict) -> float:
    """Which unsure answers deserve the slow second look first: things seen often and seen large."""
    x0, y0, x1, y1 = ob["bbox"]
    return len(o["obs_ids"]) * min(x1 - x0, y1 - y0)


def second_look_queue(
    work: list[tuple[dict, str, str, list[dict]]], cache: dict[str, dict], min_confidence: float, radius_m: float = 3.0
) -> list[tuple[dict, str, str]]:
    """Unsure answers worth a careful look, best first. Long walks see one board from many spots, so a sign within
    radius_m of one already read with confidence, or of one earlier in the queue, is the same board: skip it."""
    settled: list[np.ndarray] = []
    unsure = []
    for o, kind, question, cands in work:
        for ob in cands:
            entry = cache.get(f"{ob['obs_id']}|{kind}")
            if entry is None or len(entry["attempts"]) >= 2 or entry["confidence"] >= min_confidence:
                if entry is not None and kind == "sign" and entry["confidence"] >= min_confidence:
                    settled.append(np.asarray(ob["centroid"], float))
                continue
            unsure.append((second_priority(o, ob), ob, kind, question))
    unsure.sort(key=lambda u: -u[0])
    queue = []
    for _, ob, kind, question in unsure:
        c = np.asarray(ob["centroid"], float)
        if kind == "sign" and any(np.linalg.norm(c - x) < radius_m for x in settled):
            continue
        if kind == "sign":
            settled.append(c)
        queue.append((ob, kind, question))
    return queue


def run(session: Session, checker: VisionChecker | None, more_second_looks: bool = False) -> list[dict]:
    """Answers are cached per (observation, kind), so re-running after association changes is free.

    Without a second-look budget each check escalates on its own (quick look, careful look if unsure). With
    one (long walks: hundreds of signs, the careful look costs ~10x the quick one), every observation gets the
    quick look first, then the unsure ones get careful looks in priority order until the budget runs out.
    """
    cfg = settings(session.scene())
    budget = cfg["second_budget_s"]
    work = targets(session)
    cache_path = session.root / "attr_cache.json"
    cache: dict[str, dict] = session.read(cache_path) if cache_path.exists() else {}

    def ask(ob: dict, kind: str, question: str, mode: str) -> None:
        c = save_crops(session, ob)
        ans = checker.check(c["ctx"], question, tight=c["tight"], wide=c["wide"], mode=mode)
        key = f"{ob['obs_id']}|{kind}"
        attempts = [*cache[key]["attempts"], *ans.attempts] if mode == "second" and key in cache else ans.attempts
        if kind == "sign" and not ans.answer.get("readable", True):
            ans.confidence *= 0.3
        cache[key] = {
            "kind": kind,
            "obs_id": ob["obs_id"],
            "seen_at": ob["t"],
            "answer": ans.answer,
            "confidence": round(ans.confidence, 3),
            "attempts": attempts,
            "used_cloud": ans.used_cloud,
            "crop": str(c["ctx"].relative_to(session.root)),
        }
        session.write(cache_path, cache)

    if more_second_looks and budget is None:
        budget = 1800
    if checker is not None:
        n = sum(len(c) for *_, c in work)
        for i, (_, kind, question, cands) in enumerate(work):
            for ob in cands:
                if f"{ob['obs_id']}|{kind}" not in cache:
                    ask(ob, kind, question, "auto" if budget is None else "quick")
            if i % 25 == 0:
                log.info("attributes: %d/%d objects checked", i, len(work))
        if budget is not None:
            unsure = second_look_queue(work, cache, cfg["min_confidence"])
            t0 = time.perf_counter()
            done = 0
            for ob, kind, question in unsure:
                if time.perf_counter() - t0 > budget:
                    break
                ask(ob, kind, question, "second")
                done += 1
            log.info("second looks: %d of %d unsure (%d checks, budget %ds)", done, len(unsure), n, budget)

    objects = session.read(session.objects)
    by_id = {o["object_id"]: o for o in objects}
    results = []
    for t, kind, _, cands in work:
        o = by_id[t["object_id"]]
        rows = [{"object_id": o["object_id"], **cache[k]} for ob in cands if (k := f"{ob['obs_id']}|{kind}") in cache]
        if not rows:
            continue
        best = max(rows, key=lambda r: r["confidence"])
        if kind == "sign":
            o["text"] = best["answer"].get("text")
            o["text_english"] = best["answer"].get("text_english")
        else:
            o["state"] = best["answer"]
            o["state_seen_at"] = best["seen_at"]
        o["attribute_confidence"] = best["confidence"]
        o["attribute_obs"] = best["obs_id"]
        results.append(best)
        log.info(
            "%s %s: %s (conf %.2f, %s)",
            o["object_id"],
            kind,
            best["answer"],
            best["confidence"],
            "+".join(best["attempts"]),
        )
    session.write(session.objects, objects)
    session.write(session.attributes, results)
    return results
