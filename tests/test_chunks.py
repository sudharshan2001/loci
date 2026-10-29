import numpy as np

from loci.chunks import align, chunk_ranges, transform_points, transform_poses


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def test_chunk_ranges_cover_and_overlap() -> None:
    assert chunk_ranges(50, 60, 10) == [(0, 50)]
    r = chunk_ranges(130, 60, 10)
    assert r == [(0, 60), (50, 110), (100, 130)]
    assert all(b[0] < a[1] for a, b in zip(r, r[1:], strict=False))  # consecutive chunks overlap


def test_align_straight_walk_recovers_transform() -> None:
    # cameras walking in a straight line (collinear centres), with a little head turn
    rng = np.random.default_rng(0)
    world = np.stack([np.eye(4) for _ in range(10)])
    for i in range(10):
        world[i, :3, :3] = _rot(np.array([0, 1.0, 0]), 0.05 * rng.normal())
        world[i, :3, 3] = [0, 0, 1.4 * i]
    # the second chunk sees the same cameras in its own frame (rotated, shifted, slightly rescaled)
    R_true, s_true, t_true = _rot(np.array([0.3, 1.0, 0.2]), 0.8), 1.1, np.array([2.0, -0.5, 7.0])
    s_inv, R_inv = 1 / s_true, R_true.T
    t_inv = -s_inv * (R_inv @ t_true)
    chunk = transform_poses(world, s_inv, R_inv, t_inv)
    s, R, t = align(world[6:], chunk[6:])
    assert abs(s - s_true) < 1e-6 and np.allclose(R, R_true, atol=1e-6)
    back = transform_poses(chunk, s, R, t)
    assert np.allclose(back, world, atol=1e-6)
    p = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(transform_points(transform_points(p, s_inv, R_inv, t_inv), s, R, t), p, atol=1e-6)
