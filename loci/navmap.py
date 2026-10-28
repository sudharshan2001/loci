"""Robot handoff: a floor grid from the (gravity-aligned) point cloud and an A* path to an object.

World axes are OpenCV with y pointing down, so height above the floor is floor_y - y.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

FREE, UNKNOWN, OBSTACLE = 0, 1, 2
COST = {FREE: 1.0, UNKNOWN: 3.0}


@dataclass
class Grid:
    cells: np.ndarray  # (nz, nx) of FREE / UNKNOWN / OBSTACLE
    x0: float
    z0: float
    res: float
    floor_y: float

    def to_cell(self, x: float, z: float) -> tuple[int, int]:
        return int((z - self.z0) / self.res), int((x - self.x0) / self.res)

    def to_world(self, r: int, c: int) -> tuple[float, float]:
        return self.x0 + (c + 0.5) * self.res, self.z0 + (r + 0.5) * self.res


def floor_height(pts: np.ndarray, cams: np.ndarray, eye_height: float = 1.5) -> float:
    """Floor y: the densest horizontal layer near eye-height below the cameras, else the eye-height guess."""
    guess = float(np.median(cams[:, 1]) + eye_height)
    near = pts[np.abs(pts[:, 1] - guess) < 0.5, 1]
    if len(near) < 500:
        return guess
    hist, edges = np.histogram(near, bins=50)
    k = int(np.argmax(hist))
    return float((edges[k] + edges[k + 1]) / 2)


def build_grid(
    pts: np.ndarray, cams: np.ndarray, res: float = 0.05, robot_radius: float = 0.2, margin: float = 0.5
) -> Grid:
    floor_y = floor_height(pts, cams)
    h = floor_y - pts[:, 1]  # height above floor
    lo = np.minimum(pts[:, [0, 2]].min(0), cams[:, [0, 2]].min(0)) - margin
    hi = np.maximum(pts[:, [0, 2]].max(0), cams[:, [0, 2]].max(0)) + margin
    nx, nz = (np.ceil((hi - lo) / res).astype(int) + 1).tolist()
    cells = np.full((nz, nx), UNKNOWN, np.uint8)

    def idx(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return ((p[:, 1] - lo[1]) / res).astype(int), ((p[:, 0] - lo[0]) / res).astype(int)

    floor = pts[np.abs(h) < 0.08]
    r, c = idx(floor[:, [0, 2]])
    cells[r, c] = FREE
    # where the person walked is traversable
    r, c = idx(cams[:, [0, 2]])
    for rr, cc in zip(r, c, strict=True):
        cells[max(0, rr - 2) : rr + 3, max(0, cc - 2) : cc + 3] = FREE
    obst = pts[(h > 0.15) & (h < 1.6)]
    counts = np.zeros((nz, nx), np.int32)
    r, c = idx(obst[:, [0, 2]])
    np.add.at(counts, (r, c), 1)
    blocked = (counts >= 3).astype(np.uint8)
    k = max(1, int(round(robot_radius / res)))
    blocked = cv2.dilate(blocked, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    cells[blocked.astype(bool)] = OBSTACLE
    return Grid(cells, float(lo[0]), float(lo[1]), res, floor_y)


def astar(cells: np.ndarray, start: tuple[int, int], goals: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """8-connected A* over FREE/UNKNOWN cells to the nearest goal cell. Empty list if unreachable."""
    if not goals:
        return []
    garr = np.array(list(goals))

    def heur(p: tuple[int, int]) -> float:
        return float(np.min(np.hypot(garr[:, 0] - p[0], garr[:, 1] - p[1])))

    nz, nx = cells.shape
    openq = [(heur(start), 0.0, start)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    gcost = {start: 0.0}
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    while openq:
        _, g, cur = heapq.heappop(openq)
        if cur in goals:
            path = [cur]
            while path[-1] in came:
                path.append(came[path[-1]])
            return path[::-1]
        if g > gcost.get(cur, np.inf):
            continue
        for dr, dc in steps:
            nb = (cur[0] + dr, cur[1] + dc)
            if not (0 <= nb[0] < nz and 0 <= nb[1] < nx) or cells[nb] == OBSTACLE:
                continue
            ng = g + COST[int(cells[nb])] * (1.4142 if dr and dc else 1.0)
            if ng < gcost.get(nb, np.inf):
                gcost[nb] = ng
                came[nb] = cur
                heapq.heappush(openq, (ng + heur(nb), ng, nb))
    return []


def nearest_open(cells: np.ndarray, rc: tuple[int, int]) -> tuple[int, int] | None:
    free = np.argwhere(cells != OBSTACLE)
    if len(free) == 0:
        return None
    k = int(np.argmin(np.hypot(free[:, 0] - rc[0], free[:, 1] - rc[1])))
    return int(free[k, 0]), int(free[k, 1])


def plan(
    grid: Grid, start_xz: tuple[float, float], box_min: list[float], box_max: list[float], reach_m: float = 0.6
) -> dict[str, Any]:
    """Path from start to a standing spot within reach of the object's footprint."""
    cells = grid.cells
    start = nearest_open(cells, grid.to_cell(*start_xz))
    if start is None:
        return {"ok": False, "reason": "no open floor"}
    (x0, _, z0), (x1, _, z1) = box_min, box_max
    rr, cc = np.mgrid[0 : cells.shape[0], 0 : cells.shape[1]]
    wx, wz = grid.x0 + (cc + 0.5) * grid.res, grid.z0 + (rr + 0.5) * grid.res
    dx = np.maximum(np.maximum(x0 - wx, wx - x1), 0)
    dz = np.maximum(np.maximum(z0 - wz, wz - z1), 0)
    near = (np.hypot(dx, dz) <= reach_m) & (cells != OBSTACLE)
    goals = {(int(r), int(c)) for r, c in np.argwhere(near)}
    path = astar(cells, start, goals)
    if not path:
        return {"ok": False, "reason": "no path found"}
    pts = np.array([grid.to_world(r, c) for r, c in path])
    # keep every ~0.25 m and the end
    keep = [0]
    acc = 0.0
    for i in range(1, len(pts)):
        acc += float(np.hypot(*(pts[i] - pts[i - 1])))
        if acc >= 0.25 or i == len(pts) - 1:
            keep.append(i)
            acc = 0.0
    way = [[round(float(x), 2), round(grid.floor_y, 2), round(float(z), 2)] for x, z in pts[keep]]
    length = float(np.sum(np.hypot(*np.diff(pts, axis=0).T))) if len(pts) > 1 else 0.0
    unknown = float(np.mean([cells[r, c] == UNKNOWN for r, c in path]))
    return {
        "ok": True,
        "waypoints_m": way,
        "length_m": round(length, 2),
        "approach_point_m": way[-1],
        "through_unscanned_fraction": round(unknown, 2),
    }
