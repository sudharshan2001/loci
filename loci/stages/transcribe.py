"""Stage 10: transcribe the capture's audio and link each utterance to objects in view within +/- window."""

from __future__ import annotations

import logging
from dataclasses import asdict

from loci.providers.base import Transcriber
from loci.session import Session

log = logging.getLogger(__name__)


def link(utterances: list[dict], observations: list[dict], window_s: float = 2.0) -> list[dict]:
    """For each utterance, objects observed within its time span +/- window, nearest in time first."""
    links = []
    for i, u in enumerate(utterances):
        lo, hi = u["start_s"] - window_s, u["end_s"] + window_s
        seen: dict[str, float] = {}
        for ob in observations:
            if lo <= ob["t"] <= hi:
                mid = (u["start_s"] + u["end_s"]) / 2
                d = abs(ob["t"] - mid)
                oid = ob["object_id"]
                seen[oid] = min(d, seen.get(oid, d))
        for oid, d in sorted(seen.items(), key=lambda x: x[1]):
            links.append({"utterance": i, "object_id": oid, "dt_s": round(d, 2)})
    return links


def run(session: Session, transcriber: Transcriber) -> dict:
    if not session.audio.exists():
        result = {"utterances": [], "links": []}
        session.write(session.utterances, result)
        return result
    utts = [asdict(u) for u in transcriber.transcribe(session.audio)]
    obj_of = {oid: o["object_id"] for o in session.read(session.objects) for oid in o["obs_ids"]}
    obs = [
        {**ob, "object_id": obj_of[ob["obs_id"]]}
        for ob in session.read(session.observations)
        if ob["obs_id"] in obj_of
    ]
    result = {"utterances": utts, "links": link(utts, obs)}
    session.write(session.utterances, result)
    log.info("%s: %d utterances, %d links", session.id, len(utts), len(result["links"]))
    return result
