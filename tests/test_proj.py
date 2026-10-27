import numpy as np

from loci.proj import backproject, inliers, map_box, map_mask, preproc_map, robust_box


def test_preproc_map_landscape_1080p() -> None:
    pm = preproc_map(1920, 1080, 518, 294)
    assert pm.scale < 1
    # scaled size covers the target, crop is centred
    assert int(1920 * pm.scale) >= 518 and int(1080 * pm.scale) >= 294
    assert pm.left >= 0 and pm.top >= 0


def test_preproc_map_no_upscale() -> None:
    pm = preproc_map(400, 300, 518, 392)
    assert pm.scale == 1.0


def test_map_mask_matches_box() -> None:
    mask = np.zeros((1080, 1920), bool)
    mask[400:600, 800:1100] = True
    pm = preproc_map(1920, 1080, 518, 294)
    m = map_mask(mask, pm)
    assert m.shape == (294, 518)
    ys, xs = np.nonzero(m)
    bx = map_box([800, 400, 1100, 600], pm)
    assert abs(xs.min() - bx[0]) <= 1 and abs(ys.max() + 1 - bx[3]) <= 1


def test_backproject_plane_identity_pose() -> None:
    K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    depth = np.full((80, 100), 2.0)
    mask = np.zeros_like(depth, bool)
    mask[39:42, 49:52] = True  # pixels around the principal point
    pts = backproject(depth, K, np.eye(4), mask)
    assert np.allclose(pts[:, 2], 2.0)
    assert np.abs(pts[:, :2]).max() < 0.03


def test_backproject_with_pose() -> None:
    K = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    depth = np.full((80, 100), 1.0)
    mask = np.zeros_like(depth, bool)
    mask[40, 50] = True  # the principal point
    pose = np.eye(4)
    pose[:3, 3] = [1.0, 2.0, 3.0]  # camera moved, no rotation
    pts = backproject(depth, K, pose, mask)
    assert np.allclose(pts[0], [1.0, 2.0, 4.0], atol=1e-6)
    # 90 degrees about y: camera +z maps to world +x
    rot = np.eye(4)
    rot[:3, :3] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    pts = backproject(depth, K, rot, mask)
    assert np.allclose(pts[0], [1.0, 0.0, 0.0], atol=1e-6)


def test_inliers_and_box_ignore_outliers() -> None:
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 0.05, (500, 3))
    pts[:5] = 50.0
    keep = inliers(pts)
    assert not keep[:5].any() and keep[5:].mean() > 0.95
    c, lo, hi = robust_box(pts[keep])
    assert np.abs(c).max() < 0.02 and (hi - lo).max() < 0.3


def test_plane_up_finds_tilted_floor() -> None:
    from loci.stages.geometry import plane_up

    rng = np.random.default_rng(1)
    true_up = np.array([0.1, -0.99, 0.05])
    true_up /= np.linalg.norm(true_up)
    e1 = np.cross(true_up, [1.0, 0, 0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(true_up, e1)
    floor = rng.uniform(-2, 2, (3000, 1)) * e1 + rng.uniform(-2, 2, (3000, 1)) * e2
    clutter = rng.uniform(-2, 2, (1000, 3))
    prior = np.array([0.4, -0.9, -0.2])  # biased camera estimate
    up, frac = plane_up(np.vstack([floor, clutter]), prior / np.linalg.norm(prior))
    assert up @ true_up > 0.999 and frac > 0.5


def test_rotation_between() -> None:
    from loci.stages.geometry import rotation_between

    a = np.array([0.098, -0.911, -0.4])
    a /= np.linalg.norm(a)
    R = rotation_between(a, np.array([0.0, -1.0, 0.0]))
    assert np.allclose(R @ a, [0, -1, 0], atol=1e-6) and np.allclose(R @ R.T, np.eye(3), atol=1e-6)
    R = rotation_between(np.array([0.0, 1.0, 0.0]), np.array([0.0, -1.0, 0.0]))
    assert np.allclose(R @ [0, 1.0, 0], [0, -1, 0], atol=1e-6)
