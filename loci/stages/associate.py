"""Stage 7: merge observations of the same physical thing into objects.

Greedy, highest score first: an observation joins an existing object with a compatible label (same label,
or one is the other's alias) when their 3D boxes overlap or their centroids are closer than a size-aware
threshold. Otherwise it starts a new object.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np

from loci.session import Session

log = logging.getLogger(__name__)


def box_iou(a_min, a_max, b_min, b_max) -> float:
    a_min, a_max, b_min, b_max = map(np.asarray, (a_min, a_max, b_min, b_max))
    inter = np.clip(np.minimum(a_max, b_max) - np.maximum(a_min, b_min), 0, None).prod()
    va, vb = np.prod(a_max - a_min), np.prod(b_max - b_min)
    return float(inter / (va + vb - inter)) if va + vb - inter > 0 else 0.0


def is_sign(label: str) -> bool:
    return label.endswith("sign") or label.endswith("signboard")


def labels_compatible(a: dict, b: dict) -> bool:
    la = {a["label"], *a.get("aliases", [])}
    lb = {b["label"], *b.get("aliases", [])}
    return bool(la & lb)


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")


class _Obj:
    def __init__(self, ob: dict) -> None:
        self.members = [ob]
        self.keyframes = {ob.get("keyframe")}
        self.label = ob["label"]
        self.aliases: set[str] = set(ob.get("aliases", []))
        self._update()

    def _update(self) -> None:
        # each view sees part of an object, so the box is the union of views (robust percentiles)
        c = np.array([m["centroid"] for m in self.members])
        lo = np.array([m["box_min"] for m in self.members])
        hi = np.array([m["box_max"] for m in self.members])
        self.centroid = np.median(c, 0)
        q = 10 if len(self.members) >= 5 else 0
        self.lo = np.minimum(np.percentile(lo, q, 0), self.centroid)
        self.hi = np.maximum(np.percentile(hi, 100 - q, 0), self.centroid)
        self.half = (self.hi - self.lo) / 2

    def as_match(self) -> dict:
        return {"label": self.label, "aliases": list(self.aliases)}

    def matches(self, ob: dict, min_dist: float, iou_thresh: float) -> float | None:
        """Return a distance score if ob belongs to this object, else None."""
        if not labels_compatible(self.as_match(), ob):
            return None
        # two boards in one frame are two boards: without this, boxes grow along a row of shops until one "sign"
        # holds a dozen of them and only one is ever read
        if is_sign(self.label) and ob.get("keyframe") in self.keyframes:
            return None
        oc = np.array(ob["centroid"])
        d = float(np.linalg.norm(oc - self.centroid))
        size = float(np.linalg.norm(self.half) + np.linalg.norm((np.array(ob["box_max"]) - ob["box_min"]) / 2))
        if d <= max(min_dist, 0.6 * size):
            return d
        if box_iou(self.lo, self.hi, ob["box_min"], ob["box_max"]) >= iou_thresh:
            return d
        return None

    def add(self, ob: dict) -> None:
        self.members.append(ob)
        self.keyframes.add(ob.get("keyframe"))
        self.aliases |= set(ob.get("aliases", []))
        if ob["label"] != self.label:
            self.aliases.add(ob["label"])
        self._update()


def merge_overlapping(objs: list[_Obj]) -> list[_Obj]:
    """Second pass: merge objects with compatible labels whose union boxes intersect.

    Partial views of one big thing (a bed, a chair at the frame edge) start separate objects in the greedy pass.
    """
    merged = True
    while merged:
        merged = False
        for i, a in enumerate(objs):
            for b in objs[i + 1 :]:
                if is_sign(a.label) and a.keyframes & b.keyframes:
                    continue
                if labels_compatible(a.as_match(), b.as_match()) and box_iou(a.lo, a.hi, b.lo, b.hi) > 0.05:
                    for m in b.members:
                        a.add(m)
                    objs.remove(b)
                    merged = True
                    break
            if merged:
                break
    return objs


def associate(
    observations: list[dict], min_dist: float = 0.25, iou_thresh: float = 0.15, single_min_score: float = 0.6
) -> list[dict[str, Any]]:
    objs: list[_Obj] = []
    for ob in sorted(observations, key=lambda o: -o["score"]):
        best, best_d = None, None
        for o in objs:
            d = o.matches(ob, min_dist, iou_thresh)
            if d is not None and (best_d is None or d < best_d):
                best, best_d = o, d
        if best is None:
            objs.append(_Obj(ob))
        else:
            best.add(ob)
    objs = merge_overlapping(objs)
    # a thing seen in one frame only needs a confident detection to count
    objs = [o for o in objs if len(o.members) > 1 or o.members[0]["score"] >= single_min_score]

    out = []
    counters: dict[str, int] = {}
    for o in sorted(objs, key=lambda o: min(m["t"] for m in o.members)):
        k = counters[slug(o.label)] = counters.get(slug(o.label), 0) + 1
        ms = o.members
        best = max(ms, key=lambda m: m["quality"])
        areas = [m["area_label"] for m in ms]
        n = len(ms)
        out.append(
            {
                "object_id": f"{slug(o.label)}_{k}",
                "label": o.label,
                "aliases": sorted(o.aliases - {o.label}),
                "centroid": o.centroid.round(4).tolist(),
                "box_min": o.lo.round(4).tolist(),
                "box_max": o.hi.round(4).tolist(),
                "extent_m": (2 * o.half).round(3).tolist(),
                "first_seen": min(m["t"] for m in ms),
                "last_seen": max(m["t"] for m in ms),
                "n_obs": n,
                "score": round(float(np.mean([m["score"] for m in ms])), 3),
                # one-frame detections are less trustworthy
                "confidence": round(float(np.mean([m["score"] for m in ms]) * min(1.0, 0.5 + n / 6)), 3),
                "area": max(set(areas), key=areas.count),
                "best_obs": best["obs_id"],
                "obs_ids": [m["obs_id"] for m in sorted(ms, key=lambda m: m["t"])],
            }
        )
    return out


def run(session: Session) -> list[dict]:
    objs = associate(session.read(session.observations))
    session.write(session.objects, objs)
    log.info("%s: %d objects", session.id, len(objs))
    return objs
