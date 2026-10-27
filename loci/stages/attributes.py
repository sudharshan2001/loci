"""Stage 9: Glimmer reads signs and checks the state of stateful objects, from redacted crops.

Crops (tight / context / wide) go to <session>/crops/ and are reused as evidence. A state is stored with the
timestamp of the observation it came from: answers say "as of <time>", never "now".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2

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
    (("door",), 'Is this door open or closed? Return JSON {"door_open": true/false}.'),
    (("window",), 'Is this window open or closed? Return JSON {"window_open": true/false}.'),
    (
        ("tv", "television", "monitor", "screen"),
        "Is this screen switched on (lit, showing anything)? The content may be blurred for privacy; "
        'judge only whether it is lit. Return JSON {"screen_on": true/false}.',
    ),
    (("fan",), 'Is this fan running (blurred blades)? Return JSON {"fan_on": true/false}.'),
]


def question_for(label: str) -> tuple[str, str] | None:
    """('sign' | 'state', question) for labels that need a visual attribute check."""
    if label in SIGN_LABELS or label.endswith(" sign") or label.endswith("signboard"):
        return "sign", SIGN_Q
    for keys, q in STATE_QS:
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


def pending(session: Session, per_object: int = 2) -> int:
    """How many vision checks are not yet cached (0 means no Glimmer needed)."""
    cache_path = session.root / "attr_cache.json"
    cache = session.read(cache_path) if cache_path.exists() else {}
    obs = {o["obs_id"]: o for o in session.read(session.observations)}
    n = 0
    for o in session.read(session.objects):
        q = question_for(o["label"])
        if q:
            cands = sorted((obs[i] for i in o["obs_ids"]), key=lambda m: -m["quality"])[:per_object]
            n += sum(f"{c['obs_id']}|{q[0]}" not in cache for c in cands)
    return n


def run(session: Session, checker: VisionChecker | None, per_object: int = 2) -> list[dict]:
    """Answers are cached per (observation, kind), so re-running after association changes is free."""
    objects = session.read(session.objects)
    obs = {o["obs_id"]: o for o in session.read(session.observations)}
    cache_path = session.root / "attr_cache.json"
    cache: dict[str, dict] = session.read(cache_path) if cache_path.exists() else {}
    results = []
    for o in objects:
        q = question_for(o["label"])
        if q is None:
            continue
        kind, question = q
        cands = sorted((obs[i] for i in o["obs_ids"]), key=lambda m: -m["quality"])[:per_object]
        best: dict[str, Any] | None = None
        for ob in cands:
            key = f"{ob['obs_id']}|{kind}"
            if key not in cache:
                if checker is None:  # cached-only run
                    continue
                c = save_crops(session, ob)
                ans = checker.check(c["ctx"], question, tight=c["tight"], wide=c["wide"])
                if kind == "sign" and not ans.answer.get("readable", True):
                    ans.confidence *= 0.3
                cache[key] = {
                    "kind": kind,
                    "obs_id": ob["obs_id"],
                    "seen_at": ob["t"],
                    "answer": ans.answer,
                    "confidence": round(ans.confidence, 3),
                    "attempts": ans.attempts,
                    "used_cloud": ans.used_cloud,
                    "crop": str(c["ctx"].relative_to(session.root)),
                }
                session.write(cache_path, cache)
            row = {"object_id": o["object_id"], **cache[key]}
            if best is None or row["confidence"] > best["confidence"]:
                best = row
        if best is None:
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
    session.write(session.objects, objects)
    session.write(session.attributes, results)
    return results
