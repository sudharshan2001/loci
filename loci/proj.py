"""Pure geometry: MapAnything's image preprocessing as a pixel mapping, back-projection, robust stats.

MapAnything's load_images (fixed_mapping) scales an image by s = max(tw / W, th / H) (only when s < 1),
floors the size, then center-crops to (tw, th). PreprocMap replicates that so SAM masks drawn on the
original keyframe land on the right depth pixels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocMap:
    src_w: int
    src_h: int
    scale: float
    left: int
    top: int
    out_w: int
    out_h: int

    def as_dict(self) -> dict:
        return asdict(self)


def preproc_map(src_w: int, src_h: int, out_w: int, out_h: int) -> PreprocMap:
    s = max(out_w / src_w, out_h / src_h) + 1e-8
    if s >= 1:  # MapAnything does not upscale (force=False)
        s = 1.0
        sw, sh = src_w, src_h
    else:
        sw, sh = int(np.floor(src_w * s)), int(np.floor(src_h * s))
    return PreprocMap(src_w, src_h, s, (sw - out_w) // 2, (sh - out_h) // 2, out_w, out_h)


def map_mask(mask: np.ndarray, pm: PreprocMap) -> np.ndarray:
    """Resize a source-resolution mask the way MapAnything resized the image, then crop."""
    sw, sh = (pm.src_w, pm.src_h) if pm.scale == 1.0 else (int(pm.src_w * pm.scale), int(pm.src_h * pm.scale))
    m = cv2.resize(mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    return m[pm.top : pm.top + pm.out_h, pm.left : pm.left + pm.out_w].astype(bool)


def map_box(box: list[float], pm: PreprocMap) -> list[float]:
    """Source-pixel box [x0, y0, x1, y1] to depth-map pixels."""
    x0, y0, x1, y1 = box
    return [x0 * pm.scale - pm.left, y0 * pm.scale - pm.top, x1 * pm.scale - pm.left, y1 * pm.scale - pm.top]


def backproject(depth: np.ndarray, K: np.ndarray, cam2world: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """World points (N, 3) for masked pixels of a z-depth map.

    OpenCV camera axes and integer pixel coordinates, matching MapAnything's depthmap_to_camera_coordinates.
    """
    v, u = np.nonzero(mask & (depth > 0))
    z = depth[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], 1)
    return pts @ cam2world[:3, :3].T + cam2world[:3, 3]


def inliers(pts: np.ndarray, k: float = 3.0) -> np.ndarray:
    """Boolean mask of points within k robust standard deviations of the median, per axis."""
    if len(pts) < 5:
        return np.ones(len(pts), bool)
    med = np.median(pts, 0)
    mad = np.median(np.abs(pts - med), 0) * 1.4826 + 1e-6
    return (np.abs(pts - med) <= k * mad).all(1)


def robust_box(pts: np.ndarray, lo: float = 5, hi: float = 95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(centroid, box_min, box_max) from percentiles, so stray depth points do not inflate the box."""
    return np.median(pts, 0), np.percentile(pts, lo, 0), np.percentile(pts, hi, 0)
