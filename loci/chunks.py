"""Long walks: reconstruct overlapping chunks of keyframes and join them on their shared frames.

MapAnything fits ~80 views in 16 GB. A 6-minute walk needs ~360 keyframes, so the walk is split into chunks
that overlap by a few frames; each chunk is mapped into the previous one's frame by a similarity transform
from the shared cameras. Rotation comes from the shared camera orientations (positions alone are nearly
collinear when walking straight, which leaves the roll about the walking direction undetermined).
"""

from __future__ import annotations

import numpy as np


def chunk_ranges(n: int, size: int, overlap: int) -> list[tuple[int, int]]:
    """[(start, end)) index ranges covering 0..n, each `size` long, consecutive ones sharing `overlap` frames."""
    if n <= size:
        return [(0, n)]
    step = size - overlap
    out, a = [], 0
    while True:
        b = min(a + size, n)
        out.append((a, b))
        if b == n:
            return out
        a += step


def mean_rotation(Rs: np.ndarray) -> np.ndarray:
    """Closest rotation to the average of rotation matrices (SVD projection)."""
    u, _, vt = np.linalg.svd(Rs.sum(0))
    R = u @ vt
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vt
    return R


def align(
    ref_poses: np.ndarray, cur_poses: np.ndarray, keep_scale: bool = False
) -> tuple[float, np.ndarray, np.ndarray]:
    """(s, R, t) mapping the current chunk's frame into the reference frame, from the same cameras' poses.

    ref_poses and cur_poses: (k, 4, 4) cam-to-world poses of the shared frames in each frame. keep_scale: s = 1.
    Both chunks are already metric, and a scale from a few metres of shared path is noise (on a 6-minute walk it
    shrank the last chunks by a third, while their own metric scale gave a steady walking speed).
    """
    R = mean_rotation(np.einsum("kij,klj->kil", ref_poses[:, :3, :3], cur_poses[:, :3, :3]))
    c_ref, c_cur = ref_poses[:, :3, 3], cur_poses[:, :3, 3]
    a, b = c_ref - c_ref.mean(0), (c_cur - c_cur.mean(0)) @ R.T
    spread = float(np.sqrt((b**2).sum()))
    s = float(np.sqrt((a**2).sum()) / spread) if spread > 0.05 else 1.0
    s = 1.0 if keep_scale else float(np.clip(s, 0.5, 2.0))  # both metric: only modest disagreement is plausible
    t = c_ref.mean(0) - s * (R @ c_cur.mean(0))
    return s, R, t


def transform_poses(poses: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    out = poses.copy()
    out[:, :3, :3] = R @ poses[:, :3, :3]
    out[:, :3, 3] = s * poses[:, :3, 3] @ R.T + t
    return out


def transform_points(pts: np.ndarray, s: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return s * pts @ R.T + t
