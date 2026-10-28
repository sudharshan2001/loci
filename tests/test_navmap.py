import numpy as np

from loci.navmap import OBSTACLE, astar, build_grid, plan


def _room(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """4 x 4 m room with side walls, floor at y = 1.5 (y down), and a wall across z = 2 with a 1 m doorway
    at x in [3, 4]."""
    floor = np.c_[rng.uniform(0, 4, 20000), np.full(20000, 1.5), rng.uniform(0, 4, 20000)]
    wall = np.c_[rng.uniform(0, 3, 6000), rng.uniform(0.2, 1.3, 6000), np.full(6000, 2.0)]
    left = np.c_[np.full(6000, -0.05), rng.uniform(0.2, 1.3, 6000), rng.uniform(-0.5, 4.5, 6000)]
    right = np.c_[np.full(6000, 4.05), rng.uniform(0.2, 1.3, 6000), rng.uniform(-0.5, 4.5, 6000)]
    cams = np.array([[0.5, 0.0, 0.5], [1.0, 0.0, 0.8]])  # eye height 1.5 m above the floor
    return np.vstack([floor, wall, left, right]), cams


def test_build_grid_finds_floor_and_wall() -> None:
    pts, cams = _room(np.random.default_rng(0))
    g = build_grid(pts, cams)
    assert abs(g.floor_y - 1.5) < 0.05
    r, c = g.to_cell(1.0, 2.0)
    assert g.cells[r, c] == OBSTACLE
    r, c = g.to_cell(3.5, 2.0)  # the doorway stays open
    assert g.cells[r, c] != OBSTACLE


def test_plan_goes_through_the_doorway() -> None:
    pts, cams = _room(np.random.default_rng(1))
    g = build_grid(pts, cams)
    res = plan(g, (0.5, 0.5), [0.8, 0.5, 3.4], [1.2, 0.8, 3.6])  # a box behind the wall
    assert res["ok"]
    xs = [w[0] for w in res["waypoints_m"] if 1.8 < w[2] < 2.2]
    assert xs and min(xs) > 2.7  # crossed z = 2 only through the doorway
    assert res["length_m"] > 3.0


def test_astar_unreachable() -> None:
    cells = np.zeros((5, 5), np.uint8)
    cells[:, 2] = OBSTACLE
    assert astar(cells, (0, 0), {(4, 4)}) == []
