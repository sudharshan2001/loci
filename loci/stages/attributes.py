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
from rapidfuzz import fuzz

from loci.config import load_config
from loci.providers.base import VisionChecker
from loci.session import Session
from loci.stages.associate import is_sign

log = logging.getLogger(__name__)

SIGN_LABELS = {"shop sign", "signboard", "sign", "store sign", "shop name board"}

SIGN_Q = (
    "This is a crop of a shop sign. Read the shop's name exactly as written (it may be English, Kannada or "
    'both). Return JSON {"text": name as written, "text_english": English transliteration or translation, '
    '"readable": true/false}.'
)

DESCRIBE_Q = (
    'This crop was detected as "{label}"; the photo may be sideways. Describe only that object, never any person. '
    'Return JSON {{"name": two to four words a person would call it, with its main colour, e.g. "purple plastic '
    'bucket", "sells": what is on sale if it is a stall, cart or shop, as singular nouns, e.g. ["coconut"], else []}}.'
)
PERSON_WORDS = {"man", "woman", "boy", "girl", "person", "people", "child", "vendor", "shopkeeper", "customer"}

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
        "describe": cfg.get("describe", []),
        "describe_skip": cfg.get("describe_skip", []),
        "min_confidence": cfg.get("min_confidence", 0.7),
    }


def wants_description(label: str, cfg: dict[str, Any]) -> bool:
    """Which objects get a look at what they are: "all", or those whose label has one of the given words."""
    wanted = cfg.get("describe", [])
    if is_sign(label) or any(k in label for k in cfg.get("describe_skip", [])):
        return False
    return wanted == "all" or any(k in label for k in wanted)


def description_aliases(answer: dict) -> list[str]:
    """Search names from a description: "purple plastic bucket"; "coconut", "coconut stall" for what a stall sells."""
    name = str(answer.get("name") or "").lower().strip()
    out = [name] if name and len(name.split()) <= 5 and not PERSON_WORDS & set(name.split()) else []
    sells = answer.get("sells")
    for good in (sells if isinstance(sells, list) else [])[:3]:
        good = str(good).lower().strip()
        if good and len(good.split()) <= 3 and not PERSON_WORDS & set(good.split()):
            out += [good, f"{good} stall", f"{good} shop"]
    return out


def _sign_names(o: dict) -> set[str]:
    keys = ("".join(filter(str.isalnum, t.lower())) for t in (o.get("text"), o.get("text_english")) if t)
    return {k for k in keys if len(k) >= 3}


def _same_shop(a: set[str], b: set[str]) -> bool:
    return any(fuzz.ratio(x, y) >= 88 or (min(len(x), len(y)) >= 4 and (x in y or y in x)) for x in a for y in b)


def _weighted_median(values: list[float], weights: list[int]) -> float:
    order = np.argsort(values)
    cum = np.cumsum(np.asarray(weights, float)[order])
    return float(np.asarray(values, float)[order][np.searchsorted(cum, cum[-1] / 2)])


def merge_same_signs(objects: list[dict], radius_m: float = 40.0) -> list[dict]:
    """Boards that read the same within radius_m along the walk are one shop.

    Association keeps boards apart before they are read, so a shop seen from far and near is many objects: BIBA was
    17, at 173 to 189 m, and Louis Philippe on both sides of the road. Once read, they become one, placed by the
    weight of their sightings, showing the clearest reading as proof. Other readings stay searchable as aliases.
    """
    signs = sorted(
        (o for o in objects if is_sign(o["label"]) and _sign_names(o) and o.get("distance_along_walk_m") is not None),
        key=lambda o: o["distance_along_walk_m"],
    )
    groups: list[list[dict]] = []
    for o in signs:
        names = _sign_names(o)
        d = o["distance_along_walk_m"]
        for g in groups:
            if any(abs(m["distance_along_walk_m"] - d) <= radius_m and _same_shop(_sign_names(m), names) for m in g):
                g.append(o)
                break
        else:
            groups.append([o])

    gone = set()
    for g in groups:
        if len(g) < 2:
            continue
        keep = max(g, key=lambda m: m["n_obs"])
        clearest = max(g, key=lambda m: m.get("attribute_confidence") or 0)
        weights = [m["n_obs"] for m in g]
        readings = {t for m in g for t in (m.get("text"), m.get("text_english")) if t}
        centroid = [_weighted_median([m["centroid"][i] for m in g], weights) for i in range(3)]
        sides = [m.get("side") for m in g]
        side = max(set(sides), key=lambda s: sum(w for w, x in zip(weights, sides, strict=True) if x == s))
        dist = round(_weighted_median([m["distance_along_walk_m"] for m in g], weights), 1)
        keep.update(
            obs_ids=sorted({i for m in g for i in m["obs_ids"]}),
            n_obs=sum(weights),
            first_seen=min(m["first_seen"] for m in g),
            last_seen=max(m["last_seen"] for m in g),
            confidence=max(m["confidence"] for m in g),
            centroid=centroid,
            box_min=[min(a, b) for a, b in zip(keep["box_min"], centroid, strict=True)],
            box_max=[max(a, b) for a, b in zip(keep["box_max"], centroid, strict=True)],
            text=clearest.get("text"),
            text_english=clearest.get("text_english"),
            attribute_confidence=clearest.get("attribute_confidence"),
            attribute_obs=clearest.get("attribute_obs"),
            best_obs=clearest.get("attribute_obs") or clearest["best_obs"],
            aliases=sorted(
                {*keep.get("aliases", []), *readings} - {clearest.get("text"), clearest.get("text_english")}
            ),
            distance_along_walk_m=dist,
            side=side,
            passed_at=round(_weighted_median([m["passed_at"] for m in g], weights), 2),
            location_text=f"on your {side}, about {dist:.0f} m into the walk",
        )
        gone |= {id(m) for m in g if m is not keep}
    return [o for o in objects if id(o) not in gone]


def targets(session: Session) -> list[tuple[dict, str, str, list[dict]]]:
    """(object, kind, question, observations to ask about) for every object that needs a vision check."""
    scene = session.scene()
    cfg = settings(scene)
    obs = {o["obs_id"]: o for o in session.read(session.observations)}
    out = []
    exclude = load_config()["scenes"][scene].get("exclude_from_memory", [])
    for o in session.read(session.objects):
        ranked = sorted((obs[i] for i in o["obs_ids"]), key=lambda m: -m["quality"])
        q = question_for(o["label"], scene)
        if q is not None and o["label"] not in cfg["skip"]:
            out.append((o, q[0], q[1], ranked[: cfg["per_object"]]))
        if o["label"] not in exclude and wants_description(o["label"], cfg):
            out.append((o, "describe", DESCRIBE_Q.format(label=o["label"]), ranked[:1]))
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
        if kind == "describe":
            continue
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
                    ask(ob, kind, question, "auto" if budget is None and kind != "describe" else "quick")
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
        if kind == "describe":
            o["aliases"] = sorted({*o.get("aliases", []), *description_aliases(best["answer"])})
            continue
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
    if session.scene() == "street":
        n = len(objects)
        objects = merge_same_signs(objects)
        log.info("signs that read the same: %d objects -> %d", n, len(objects))
    session.write(session.objects, objects)
    session.write(session.attributes, results)
    return results
