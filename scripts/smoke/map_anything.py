"""Smoke test: MapAnything (Apache) on N views. Reports seconds, peak VRAM, metric scale and path length.

Optionally writes a coloured point cloud (PLY) of the last run for a visual check.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

log = logging.getLogger("smoke.mapanything")


def write_ply(path: Path, pts: np.ndarray, rgb: np.ndarray) -> None:
    from plyfile import PlyData, PlyElement

    v = np.empty(len(pts), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                  ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    v["x"], v["y"], v["z"] = pts.T
    v["red"], v["green"], v["blue"] = rgb.T
    PlyData([PlyElement.describe(v, "vertex")]).write(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/map-anything-apache")
    ap.add_argument("--frames", required=True, help="folder of keyframe JPEGs")
    ap.add_argument("--views", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    ap.add_argument("--ply", default=None, help="write a point cloud of the largest run here")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images

    t0 = time.perf_counter()
    model = MapAnything.from_pretrained(args.checkpoint).to("cuda").eval()
    load_s = time.perf_counter() - t0
    log.info("loaded in %.1fs, weights %.2f GB", load_s, torch.cuda.memory_allocated() / 1e9)

    files = sorted(Path(args.frames).glob("*.jpg"))
    rows = []
    for n in args.views:
        if n > len(files):
            log.info("skip %d views: only %d frames", n, len(files))
            continue
        pick = [str(files[i]) for i in np.linspace(0, len(files) - 1, n).round().astype(int)]
        views = load_images(pick)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t1 = time.perf_counter()
        with torch.no_grad():
            preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                                apply_mask=True, mask_edges=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t1
        poses = torch.cat([p["camera_poses"] for p in preds]).float().cpu().numpy()
        cams = poses[:, :3, 3]
        path_m = float(np.linalg.norm(np.diff(cams, axis=0), axis=1).sum())
        depth = torch.cat([p["depth_z"] for p in preds]).float().cpu().numpy()
        scale = float(preds[0]["metric_scaling_factor"].float().mean())
        row = {"views": n, "seconds": round(dt, 1), "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
               "hw": list(depth.shape[1:3]), "median_depth_m": round(float(np.median(depth[depth > 0])), 2),
               "camera_path_m": round(path_m, 2), "metric_scale": round(scale, 3)}
        rows.append(row)
        log.info("%s", row)

        if args.ply:
            pts = torch.cat([p["pts3d"] for p in preds]).float().cpu().numpy().reshape(-1, 3)
            mask = torch.cat([p["mask"] for p in preds]).cpu().numpy().reshape(-1).astype(bool)
            rgb = torch.cat([p["img_no_norm"] for p in preds]).float().cpu().numpy().reshape(-1, 3)
            pts, rgb = pts[mask], (rgb[mask].clip(0, 1) * 255).astype(np.uint8)
            keep = np.random.default_rng(0).choice(len(pts), min(len(pts), 500_000), replace=False)
            write_ply(Path(args.ply), pts[keep], rgb[keep])
            np.save(Path(args.ply).with_suffix(".cams.npy"), poses)
        del preds, views

    summary = {"checkpoint": args.checkpoint, "frames": args.frames, "load_s": round(load_s, 1), "runs": rows}
    Path(args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
