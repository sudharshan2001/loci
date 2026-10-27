"""Stage 5: MapAnything on the redacted keyframes -> poses, intrinsics, metric depth, scene.ply.

World frame = MapAnything's (first camera), metres. `up` is estimated as minus the mean camera y-axis
(OpenCV y points down; the phone is held roughly level while walking).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

from loci.masks import decode
from loci.proj import map_mask, preproc_map
from loci.session import Session

log = logging.getLogger(__name__)


def estimate_up(poses: np.ndarray) -> np.ndarray:
    """Rough up from the cameras: minus the mean camera y-axis. Biased when the camera pitches down."""
    down = poses[:, :3, 1].mean(0)
    return -down / (np.linalg.norm(down) + 1e-9)


def plane_up(
    pts: np.ndarray, prior: np.ndarray, max_angle_deg: float = 40, thresh: float = 0.02, iters: int = 400
) -> tuple[np.ndarray, float]:
    """Normal of the largest roughly-horizontal plane (floor, table top) via RANSAC, oriented like prior.

    Returns (up, inlier fraction). Falls back to prior when no plane is found.
    """
    rng = np.random.default_rng(0)
    if len(pts) > 60_000:
        pts = pts[rng.choice(len(pts), 60_000, replace=False)]
    cos_max = np.cos(np.radians(max_angle_deg))
    best_n, best_a, best_count = prior, None, 0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        if abs(n @ prior) < cos_max:
            continue
        count = int((np.abs((pts - a) @ n) < thresh).sum())
        if count > best_count:
            best_n, best_a, best_count = n, a, count
    if best_a is None:
        return prior, 0.0
    # refine the normal with least squares on the inliers
    inl = pts[np.abs((pts - best_a) @ best_n) < thresh]
    if len(inl) >= 3:
        _, _, vt = np.linalg.svd(inl - inl.mean(0), full_matrices=False)
        best_n = vt[2]
    best_n = best_n if best_n @ prior > 0 else -best_n
    return best_n / np.linalg.norm(best_n), best_count / len(pts)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Rotation matrix taking unit vector a onto unit vector b (Rodrigues)."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, c = np.cross(a, b), float(a @ b)
    if c < -1 + 1e-9:  # opposite: rotate 180 degrees about any axis orthogonal to a
        axis = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx / (1 + c)


def _cat(preds: list[dict], key: str) -> np.ndarray:
    import torch

    return torch.cat([p[key] for p in preds]).float().cpu().numpy()


def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    v = np.empty(
        len(pts),
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")],
    )
    v["x"], v["y"], v["z"] = pts.T
    v["red"], v["green"], v["blue"] = rgb.T
    PlyData([PlyElement.describe(v, "vertex")], text=False).write(str(path))


def run(
    session: Session,
    checkpoint: str,
    max_points: int = 500_000,
    conf_percentile: float = 20,
    street_max_depth_m: float = 25.0,
) -> dict:
    import torch
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    kfs = session.keyframes()
    files = [str(session.keyframes_dir / k["file"]) for k in kfs]
    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = MapAnything.from_pretrained(checkpoint).to("cuda").eval()
    load_s = time.perf_counter() - t0

    views = load_images(files)
    t1 = time.perf_counter()
    with torch.no_grad():
        preds = model.infer(
            views,
            memory_efficient_inference=True,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
        )
    infer_s = time.perf_counter() - t1

    poses, K = _cat(preds, "camera_poses"), _cat(preds, "intrinsics")
    depth, conf, pts3d = _cat(preds, "depth_z")[..., 0], _cat(preds, "conf"), _cat(preds, "pts3d")
    valid = torch.cat([p["mask"] for p in preds]).cpu().numpy()[..., 0].astype(bool)
    rgb = (_cat(preds, "img_no_norm").clip(0, 1) * 255).astype(np.uint8)
    n, h, w = depth.shape
    del preds, model
    torch.cuda.empty_cache()

    # mapping from keyframe pixels to depth pixels (all keyframes share one size)
    from PIL import Image

    src_w, src_h = Image.open(files[0]).size
    pm = preproc_map(src_w, src_h, w, h)
    assert (pm.out_w, pm.out_h) == (w, h)

    # confidence floor per frame
    keep = valid & (conf >= np.percentile(conf[valid], conf_percentile))
    scene = session.scene()
    if scene == "street":
        keep &= depth <= street_max_depth_m
        if session.tracks_json.exists():
            for inst in session.read(session.tracks_json)["instances"]:
                if inst["exclude"]:
                    keep[inst["keyframe"]] &= ~map_mask(decode(inst["mask"]), pm)

    # gravity-align the world: rotate so the largest horizontal plane's normal is -y (OpenCV: y down)
    prior = estimate_up(poses)
    up_ma, plane_frac = plane_up(pts3d[valid], prior)
    R = rotation_between(up_ma, np.array([0.0, -1.0, 0.0]))
    poses = poses.copy()
    poses[:, :3, :3] = R @ poses[:, :3, :3]
    poses[:, :3, 3] = poses[:, :3, 3] @ R.T
    pts3d = pts3d @ R.T
    up = np.array([0.0, -1.0, 0.0])
    tilt = np.degrees(np.arccos(np.clip(up_ma @ prior, -1, 1)))
    log.info(
        "gravity from largest horizontal plane (%.0f%% of points), %.1f deg from camera estimate",
        100 * plane_frac,
        tilt,
    )

    pts, cols = pts3d[keep], rgb[keep]
    if len(pts) > max_points:
        sel = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, cols = pts[sel], cols[sel]
    write_ply(session.ply, pts, cols)

    cams = poses[:, :3, 3]
    np.savez(
        session.geometry,
        poses=poses.astype(np.float32),
        intrinsics=K.astype(np.float32),
        depth=depth.astype(np.float16),
        conf=conf.astype(np.float16),
        valid=valid,
        pts3d=pts3d.astype(np.float16),
        up=up.astype(np.float32),
        world_from_mapanything=R.astype(np.float32),
        preproc=np.array([pm.src_w, pm.src_h, pm.scale, pm.left, pm.top, pm.out_w, pm.out_h], np.float64),
        keyframe_index=np.array([k["index"] for k in kfs]),
    )
    path_m = float(np.linalg.norm(np.diff(cams, axis=0), axis=1).sum())
    result = {
        "n_views": n,
        "depth_hw": [h, w],
        "load_s": round(load_s, 1),
        "infer_s": round(infer_s, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "camera_path_m": round(path_m, 2),
        "median_depth_m": round(float(np.median(depth[valid])), 2),
        "ply_points": int(len(pts)),
        "up": up.round(3).tolist(),
    }
    meta = session.read(session.meta)
    meta["geometry"] = result
    session.write(session.meta, meta)
    log.info("%s: %s", session.id, result)
    return result


def load(session: Session) -> dict[str, np.ndarray]:
    g = dict(np.load(session.geometry))
    p = g["preproc"]
    g["pm"] = preproc_map(int(p[0]), int(p[1]), int(p[5]), int(p[6]))
    return g
