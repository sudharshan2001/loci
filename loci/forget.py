"""Forget an object for good: memory rows, evidence, its pixels in stored frames, its 3D points.

Irreversible by design (privacy you can act on): callers must confirm.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from loci.masks import decode
from loci.session import Session
from loci.stages.redact import blur_regions

log = logging.getLogger(__name__)


def _cut_points(session: Session, box_min: list[float], box_max: list[float], pad: float = 0.03) -> int:
    from plyfile import PlyData, PlyElement

    ply = PlyData.read(str(session.ply))
    v = ply["vertex"].data
    p = np.stack([v["x"], v["y"], v["z"]], 1)
    inside = ((p >= np.asarray(box_min) - pad) & (p <= np.asarray(box_max) + pad)).all(1)
    PlyData([PlyElement.describe(v[~inside], "vertex")], text=False).write(str(session.ply))
    return int(inside.sum())


def forget(session: Session, con: sqlite3.Connection, object_id: str) -> dict[str, Any]:
    objects = session.read(session.objects)
    obj = next((o for o in objects if o["object_id"] == object_id), None)
    if obj is None:
        raise KeyError(f"no object {object_id} in {session.id}")
    obs_ids = set(obj["obs_ids"])

    # 1. blur the object's pixels in every stored (already redacted) keyframe, then drop its masks
    tracks = session.read(session.tracks_json)
    by_frame: dict[int, list[np.ndarray]] = {}
    for inst in tracks["instances"]:
        if inst["obs_id"] in obs_ids:
            by_frame.setdefault(inst["keyframe"], []).append(decode(inst["mask"]))
    for kf, masks in by_frame.items():
        path = session.keyframes_dir / f"{kf:04d}.jpg"
        img = cv2.imread(str(path))
        cv2.imwrite(str(path), blur_regions(img, np.any(masks, axis=0)), [cv2.IMWRITE_JPEG_QUALITY, 92])
    tracks["instances"] = [i for i in tracks["instances"] if i["obs_id"] not in obs_ids]
    session.write(session.tracks_json, tracks)

    # 2. evidence crops and rendered evidence images
    removed_files = 0
    for folder in ("crops", "evidence"):
        d = session.root / folder
        if d.exists():
            for f in d.iterdir():
                if any(f.name.startswith(o) for o in obs_ids):
                    f.unlink()
                    removed_files += 1

    # 3. 3D points inside the object's box (the floor grid cache is rebuilt from the new cloud)
    n_points = _cut_points(session, obj["box_min"], obj["box_max"])

    # 4. session files and the attribute cache
    session.write(session.objects, [o for o in objects if o["object_id"] != object_id])
    session.write(session.observations, [o for o in session.read(session.observations) if o["obs_id"] not in obs_ids])
    cache_path = session.root / "attr_cache.json"
    if cache_path.exists():
        session.write(cache_path, {k: v for k, v in session.read(cache_path).items() if k.split("|")[0] not in obs_ids})

    # 5. memory rows
    sid = session.id
    with con:
        for table, col in [
            ("objects", "object_id"),
            ("observations", "object_id"),
            ("relations", "subject"),
            ("relations", "object"),
            ("utterance_links", "object_id"),
        ]:
            con.execute(f"DELETE FROM {table} WHERE session_id=? AND {col}=?", (sid, object_id))  # noqa: S608
        con.execute("DELETE FROM alerts WHERE session_id=? AND object_id=?", (sid, f"{sid}:{object_id}"))

    # 6. the privacy ledger records that it was forgotten (no details beyond the label)
    ledger = session.read(session.ledger)
    entry = {
        "label": obj["label"],
        "at": datetime.now().isoformat(timespec="seconds"),
        "frames_blurred": len(by_frame),
        "points_removed": n_points,
    }
    ledger.setdefault("forgotten", []).append(entry)
    session.write(session.ledger, ledger)
    log.info(
        "%s: forgot %s (%d frames blurred, %d points, %d files)", sid, object_id, len(by_frame), n_points, removed_files
    )
    return {"object_id": f"{sid}:{object_id}", **entry, "files_removed": removed_files}
