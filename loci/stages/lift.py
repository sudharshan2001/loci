"""Stage 6: lift each SAM 3.1 detection into 3D using MapAnything's per-pixel world points."""

from __future__ import annotations

import logging

import cv2
import numpy as np

from loci.masks import decode
from loci.proj import inliers, map_mask, robust_box
from loci.session import Session
from loci.stages.geometry import load

log = logging.getLogger(__name__)


def quality(bbox: list[int], image_size: list[int], area: int, score: float) -> float:
    """Best-observation score: large, central, confident, not cut by the frame edge."""
    w, h = image_size
    x0, y0, x1, y1 = bbox
    cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
    centrality = 1 - min(1.0, 2 * np.hypot(cx - 0.5, cy - 0.5))
    edge = 0.5 if (x0 <= 2 or y0 <= 2 or x1 >= w - 2 or y1 >= h - 2) else 1.0
    size = min(1.0, np.sqrt(area / (w * h)) / 0.3)
    return round(float(score * edge * (0.4 + 0.3 * centrality + 0.3 * size)), 4)


def run(session: Session, min_points: int = 20, conf_percentile: float = 10) -> list[dict]:
    g = load(session)
    pm = g["pm"]
    conf = g["conf"].astype(np.float32)
    conf_floor = np.percentile(conf[g["valid"]], conf_percentile)
    kf_sharp = {k["index"]: k["sharpness"] for k in session.keyframes()}
    areas = session.read(session.vocab)["areas"]
    row_of = {int(k): i for i, k in enumerate(g["keyframe_index"])}

    obs = []
    dropped = 0
    for inst in session.read(session.tracks_json)["instances"]:
        if inst["exclude"]:
            continue
        r = row_of[inst["keyframe"]]
        m = map_mask(decode(inst["mask"]), pm)
        # erode to drop edge pixels where depth bleeds into the background
        if m.sum() > 200:
            m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        m &= g["valid"][r] & (conf[r] >= conf_floor)
        pts = g["pts3d"][r][m].astype(np.float32)
        if len(pts) < min_points:
            dropped += 1
            continue
        pts = pts[inliers(pts)]
        c, lo, hi = robust_box(pts)
        cam = g["poses"][r][:3, 3]
        obs.append(
            {
                **{
                    k: inst[k]
                    for k in ("obs_id", "keyframe", "t", "label", "aliases", "score", "bbox", "area")
                },
                "image_size": inst["image_size"],
                "centroid": c.round(4).tolist(),
                "box_min": lo.round(4).tolist(),
                "box_max": hi.round(4).tolist(),
                "n_points": int(len(pts)),
                "dist_m": round(float(np.linalg.norm(c - cam)), 3),
                "area_label": areas[inst["keyframe"]],
                "quality": quality(inst["bbox"], inst["image_size"], inst["area"], inst["score"])
                * (1.0 if kf_sharp.get(inst["keyframe"], 1) > 50 else 0.7),
            }
        )
    session.write(session.observations, obs)
    log.info(
        "%s: %d observations lifted, %d dropped (too few valid depth points)", session.id, len(obs), dropped
    )
    return obs
