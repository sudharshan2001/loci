"""Put two captures of the same place into one frame, using things that do not move.

Furniture and fixtures (desk, stove, door, bed...) appear in both captures. RANSAC over label-matched pairs
finds the similarity transform (rotation, translation, scale) that lines most of them up. Gravity is already
-y in both (geometry stage), so only a rotation about the vertical axis is allowed.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

STATIC = {
    "desk",
    "table",
    "bed",
    "sofa",
    "couch",
    "wardrobe",
    "door",
    "window",
    "gas stove",
    "stove",
    "shelf",
    "kitchen shelf",
    "cabinet",
    "dresser",
    "nightstand",
    "bookshelf",
    "kitchen counter",
    "counter",
    "sink",
    "refrigerator",
    "fridge",
    "washing machine",
    "tv unit",
    "dining table",
    "metal shelving unit",
}


def is_static(label: str) -> bool:
    return label in STATIC or label.split()[-1] in {
        "desk",
        "table",
        "bed",
        "door",
        "window",
        "shelf",
        "stove",
        "wardrobe",
        "cabinet",
        "counter",
        "sofa",
    }


def fit_yaw_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """(scale, R, t) with R a rotation about y, minimising |s R src + t - dst| (Umeyama in the xz plane)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    # rotation about y: angle from the xz components
    num = (a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]).sum()
    den = (a[:, 0] * b[:, 0] + a[:, 2] * b[:, 2]).sum()
    th = np.arctan2(num, den)
    c, s = np.cos(th), np.sin(th)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    ra = a @ R.T
    scale = float((ra * b).sum() / max((ra * ra).sum(), 1e-9))
    scale = float(np.clip(scale, 0.7, 1.4))  # both captures are metric; allow modest scale error only
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def register(src_objs: list[dict], dst_objs: list[dict], inlier_m: float = 0.5, iters: int = 2000) -> dict[str, Any]:
    """Transform taking src capture coordinates into dst coordinates, from static objects."""
    src = [o for o in src_objs if is_static(o["label"])]
    dst = [o for o in dst_objs if is_static(o["label"])]
    pairs = [(i, j) for i, a in enumerate(src) for j, b in enumerate(dst) if a["label"] == b["label"]]
    if len(pairs) < 3:
        return {"ok": False, "reason": f"only {len(pairs)} matching static objects", "n_inliers": 0}
    P = np.array([src[i]["centroid"] for i, _ in pairs], float)
    Q = np.array([dst[j]["centroid"] for _, j in pairs], float)
    rng = np.random.default_rng(0)
    best: tuple[int, list[int]] = (0, [])
    combos = list(itertools.combinations(range(len(pairs)), 3))
    for k in rng.permutation(len(combos))[:iters]:
        idx = list(combos[k])
        if len({pairs[i][0] for i in idx}) < 3 or len({pairs[i][1] for i in idx}) < 3:
            continue
        s, R, t = fit_yaw_similarity(P[idx], Q[idx])
        err = np.linalg.norm(s * P @ R.T + t - Q, axis=1)
        inl = list(np.flatnonzero(err < inlier_m))
        # one-to-one: keep the closest pair per source and per target
        inl.sort(key=lambda i: err[i])
        used_s: set[int] = set()
        used_d: set[int] = set()
        uniq = []
        for i in inl:
            if pairs[i][0] not in used_s and pairs[i][1] not in used_d:
                uniq.append(i)
                used_s.add(pairs[i][0])
                used_d.add(pairs[i][1])
        if len(uniq) > best[0]:
            best = (len(uniq), uniq)
    if best[0] < 3:
        return {"ok": False, "reason": "no consistent alignment of static objects", "n_inliers": best[0]}
    s, R, t = fit_yaw_similarity(P[best[1]], Q[best[1]])
    err = np.linalg.norm(s * P[best[1]] @ R.T + t - Q[best[1]], axis=1)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return {
        "ok": True,
        "transform": T.round(6).tolist(),
        "scale": round(s, 4),
        "n_inliers": best[0],
        "rms_m": round(float(np.sqrt((err**2).mean())), 3),
        "anchors": [src[pairs[i][0]]["label"] for i in best[1]],
    }


def apply(T: list[list[float]] | np.ndarray, p: list[float] | np.ndarray) -> np.ndarray:
    T = np.asarray(T, float)
    return T[:3, :3] @ np.asarray(p, float) + T[:3, 3]
