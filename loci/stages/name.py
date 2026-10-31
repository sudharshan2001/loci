"""Stage 3: Glimmer names the objects in a session and labels each keyframe's area."""

from __future__ import annotations

import logging
import re

import numpy as np

from loci.providers.base import Namer
from loci.session import Session

log = logging.getLogger(__name__)

# never tracked as objects: people (privacy) and room surfaces (not things you lose)
EXCLUDE = {
    "person",
    "man",
    "woman",
    "boy",
    "girl",
    "child",
    "kid",
    "people",
    "face",
    "hand",
    "arm",
    "leg",
    "wall",
    "floor",
    "ceiling",
    "room",
    "background",
}


def normalise(phrases: list[str], max_words: int = 4) -> list[str]:
    """Lowercase, strip articles and punctuation, drop people/surfaces and long phrases, dedupe in order."""
    seen: set[str] = set()
    out = []
    for p in phrases:
        p = re.sub(r"[^a-z0-9 \-]", " ", str(p).lower())
        p = re.sub(r"^(a|an|the|some)\s+", "", " ".join(p.split()))
        if not p or len(p.split()) > max_words or p in EXCLUDE or p.split()[-1] in EXCLUDE:
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def smooth(labels: list[str]) -> list[str]:
    """Overrule isolated labels: a sample that disagrees with both (agreeing) neighbours takes theirs. Where all three
    differ the room is not known, and "room" keeps it out of answers: naming no room beats naming the wrong one."""
    out = list(labels)
    for i in range(1, len(labels) - 1):
        before, here, after = labels[i - 1], labels[i], labels[i + 1]
        if before == after != here:
            out[i] = before
        elif before != here != after:
            out[i] = "room"
    return out


def fill_nearest(times: list[float], labelled: dict[int, str]) -> list[str]:
    """Give every keyframe the label of the nearest labelled keyframe in time."""
    idx = np.array(sorted(labelled))
    t = np.asarray(times)
    nearest = idx[np.abs(t[idx][None, :] - t[:, None]).argmin(1)]
    return [labelled[int(i)] for i in nearest]


def spread(n: int, k: int) -> list[int]:
    return sorted({int(i) for i in np.linspace(0, n - 1, min(k, n)).round()})


def label_areas(session: Session, namer: Namer, area_every: int) -> dict:
    """The area of every keyframe. A quick walk crosses a room in a few samples, so they must be close together."""
    kfs = session.keyframes()
    scene = session.scene()
    sampled = list(range(0, len(kfs), area_every))
    raw = [namer.label_area(session.keyframes_dir / kfs[i]["file"], scene) for i in sampled]
    labelled = dict(zip(sampled, smooth(raw), strict=True))
    return {
        "areas": fill_nearest([k["t"] for k in kfs], labelled),
        "area_samples": {str(i): a for i, a in zip(sampled, raw, strict=True)},
    }


def relabel_areas(session: Session, namer: Namer, area_every: int) -> dict:
    """New area labels only: the vocabulary stays, so tracking does not have to run again."""
    result = {**session.read(session.vocab), **label_areas(session, namer, area_every)}
    session.write(session.vocab, result)
    return result


def run(session: Session, namer: Namer, default_vocab: list[str], n_images: int = 8, area_every: int = 5) -> dict:
    files = [session.keyframes_dir / k["file"] for k in session.keyframes()]
    scene = session.scene()

    named = namer.name_objects([files[i] for i in spread(len(files), n_images)], scene)
    vocab = normalise(list(default_vocab) + named)
    result = {"scene": scene, "named": named, "vocab": vocab, **label_areas(session, namer, area_every)}
    session.write(session.vocab, result)
    areas = result["areas"]
    log.info("%s: %d phrases (%d from Glimmer), areas %s", session.id, len(vocab), len(named), sorted(set(areas)))
    return result
