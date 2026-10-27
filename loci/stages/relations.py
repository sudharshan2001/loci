"""Stage 8: spatial relations and location text.

Home: on / inside / near relative to large anchor objects, left_of / right_of as seen from the best
observation's camera, distance to the door and to where the capture started.
Street: distance along the walk, side of the road, and when it was passed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from loci.session import Session
from loci.stages.geometry import load

log = logging.getLogger(__name__)

ANCHORS = {
    "desk",
    "table",
    "shelf",
    "bed",
    "sofa",
    "couch",
    "drawer",
    "door",
    "gas stove",
    "stove",
    "cabinet",
    "wardrobe",
    "dresser",
    "nightstand",
    "bedside table",
    "counter",
    "kitchen counter",
    "chair",
    "tv unit",
    "window sill",
    "dining table",
    "coffee table",
    "bookshelf",
}


def basis(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two horizontal unit vectors orthogonal to up."""
    a = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 0, 1.0])
    e1 = np.cross(up, a)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(up, e1)


def corners(lo, hi) -> np.ndarray:
    lo, hi = np.asarray(lo), np.asarray(hi)
    return np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])


def vertical_span(o: dict, up: np.ndarray) -> tuple[float, float]:
    h = corners(o["box_min"], o["box_max"]) @ up
    return float(h.min()), float(h.max())


def footprint(o: dict, e1: np.ndarray, e2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    c = corners(o["box_min"], o["box_max"])
    p = np.stack([c @ e1, c @ e2], 1)
    return p.min(0), p.max(0)


def is_on(obj: dict, anchor: dict, up: np.ndarray, tol: float = 0.15, margin: float = 0.08) -> bool:
    e1, e2 = basis(up)
    o_lo, _ = vertical_span(obj, up)
    _, a_hi = vertical_span(anchor, up)
    c = np.asarray(obj["centroid"])
    cxy = np.array([c @ e1, c @ e2])
    f_lo, f_hi = footprint(anchor, e1, e2)
    inside_xy = bool(((cxy >= f_lo - margin) & (cxy <= f_hi + margin)).all())
    return inside_xy and abs(o_lo - a_hi) <= tol


def is_inside(obj: dict, anchor: dict) -> bool:
    c = np.asarray(obj["centroid"])
    return bool(((c >= np.asarray(anchor["box_min"])) & (c <= np.asarray(anchor["box_max"]))).all())


def side_of(point: np.ndarray, ref: np.ndarray, cam2world: np.ndarray) -> str:
    """'left_of' or 'right_of': point relative to ref, seen from the camera."""
    w2c = np.linalg.inv(cam2world)
    px = (w2c[:3, :3] @ point + w2c[:3, 3])[0]
    rx = (w2c[:3, :3] @ ref + w2c[:3, 3])[0]
    return "left_of" if px < rx else "right_of"


def home_relations(
    objects: list[dict], obs_by_id: dict, poses: np.ndarray, row_of: dict, up: np.ndarray, near_m: float = 1.0
) -> list[dict]:
    anchors = [o for o in objects if o["label"] in ANCHORS]
    rels = []
    start = poses[0][:3, 3]
    doors = [o for o in objects if o["label"] == "door"]
    for o in objects:
        c = np.asarray(o["centroid"])
        o["dist_to_start_m"] = round(float(np.linalg.norm(c - start)), 2)
        if doors:
            o["dist_to_door_m"] = round(min(float(np.linalg.norm(c - np.asarray(d["centroid"]))) for d in doors), 2)
        if o["label"] in ANCHORS:
            continue
        cam = poses[row_of[obs_by_id[o["best_obs"]]["keyframe"]]]
        for a in anchors:
            ac = np.asarray(a["centroid"])
            d = float(np.linalg.norm(c - ac))
            if is_on(o, a, up):
                rels.append({"subject": o["object_id"], "relation": "on", "object": a["object_id"], "dist_m": d})
            elif is_inside(o, a) and a["label"] in {"drawer", "shelf", "cabinet", "wardrobe", "bookshelf"}:
                rels.append({"subject": o["object_id"], "relation": "inside", "object": a["object_id"], "dist_m": d})
            elif d <= near_m:
                rels.append({"subject": o["object_id"], "relation": "near", "object": a["object_id"], "dist_m": d})
                rels.append(
                    {
                        "subject": o["object_id"],
                        "relation": side_of(c, ac, cam),
                        "object": a["object_id"],
                        "dist_m": d,
                    }
                )
    for r in rels:
        r["dist_m"] = round(r["dist_m"], 2)
    return rels


def location_text(o: dict, rels: list[dict], by_id: dict) -> str:
    mine = [r for r in rels if r["subject"] == o["object_id"]]
    parts = []
    for kind in ("on", "inside"):
        hit = min((r for r in mine if r["relation"] == kind), key=lambda r: r["dist_m"], default=None)
        if hit:
            parts.append(f"{'on' if kind == 'on' else 'inside'} the {by_id[hit['object']]['label']}")
            break
    if not parts:
        near = min((r for r in mine if r["relation"] == "near"), key=lambda r: r["dist_m"], default=None)
        if near:
            side = next(
                (
                    r["relation"]
                    for r in mine
                    if r["object"] == near["object"] and r["relation"] in ("left_of", "right_of")
                ),
                None,
            )
            where = {"left_of": "left of", "right_of": "right of"}.get(side, "near")
            parts.append(f"{where} the {by_id[near['object']]['label']}")
    if o.get("area") and o["area"] not in ("room", "street"):
        parts.append(f"in the {o['area']}")
    return ", ".join(parts) or "seen in this capture"


def walk_metrics(objects: list[dict], poses: np.ndarray, times: list[float], up: np.ndarray) -> None:
    """Street: distance along the walk, side of road, time passed (closest camera, horizontal distance)."""
    cams = poses[:, :3, 3]
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(cams, axis=0), axis=1))]
    flat = cams - np.outer(cams @ up, up)
    for o in objects:
        c = np.asarray(o["centroid"])
        cf = c - (c @ up) * up
        i = int(np.argmin(np.linalg.norm(flat - cf, axis=1)))
        j0, j1 = max(0, i - 2), min(len(cams) - 1, i + 2)
        heading = flat[j1] - flat[j0]
        if np.linalg.norm(heading) < 1e-6:  # standing still: use the camera's forward axis
            heading = poses[i][:3, 2] - (poses[i][:3, 2] @ up) * up
        to_obj = cf - flat[i]
        # up x heading points left
        o["side"] = "left" if np.cross(heading, to_obj) @ up > 0 else "right"
        o["distance_along_walk_m"] = round(float(seg[i]), 1)
        o["passed_at"] = round(float(times[i]), 2)
        o["dist_from_path_m"] = round(float(np.linalg.norm(to_obj)), 1)


def run(session: Session) -> dict[str, Any]:
    g = load(session)
    objects = session.read(session.objects)
    obs_by_id = {o["obs_id"]: o for o in session.read(session.observations)}
    row_of = {int(k): i for i, k in enumerate(g["keyframe_index"])}
    up = g["up"].astype(float)
    by_id = {o["object_id"]: o for o in objects}
    rels: list[dict] = []
    if session.scene() == "street":
        times = [k["t"] for k in session.keyframes()]
        walk_metrics(objects, g["poses"], times, up)
        for o in objects:
            o["location_text"] = f"on your {o['side']}, about {o['distance_along_walk_m']:.0f} m into the walk"
    else:
        rels = home_relations(objects, obs_by_id, g["poses"], row_of, up)
        for o in objects:
            o["location_text"] = location_text(o, rels, by_id)
    session.write(session.objects, objects)
    session.write(session.root / "relations.json", rels)
    log.info("%s: %d relations", session.id, len(rels))
    return {"objects": objects, "relations": rels}
