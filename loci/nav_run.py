"""Floor grid per session (cached on disk) and robot paths to objects."""

from __future__ import annotations

from typing import Any

import numpy as np

from loci.navmap import Grid, build_grid, plan
from loci.session import Session


def grid_for(session: Session) -> Grid:
    path = session.root / "navgrid.npz"
    if path.exists() and path.stat().st_mtime >= session.ply.stat().st_mtime:
        d = np.load(path)
        return Grid(d["cells"], float(d["x0"]), float(d["z0"]), float(d["res"]), float(d["floor_y"]))
    from plyfile import PlyData

    v = PlyData.read(str(session.ply))["vertex"]
    pts = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    cams = np.load(session.geometry)["poses"][:, :3, 3]
    g = build_grid(pts, cams)
    np.savez_compressed(path, cells=g.cells, x0=g.x0, z0=g.z0, res=g.res, floor_y=g.floor_y)
    return g


def navigate(session: Session, object_id: str) -> dict[str, Any]:
    """Path from 'you are here' (the last camera) to a standing spot beside the object."""
    obj = next((o for o in session.read(session.objects) if o["object_id"] == object_id), None)
    if obj is None:
        return {"ok": False, "reason": f"no object {object_id}"}
    here = np.load(session.geometry)["poses"][-1][:3, 3]
    res = plan(grid_for(session), (float(here[0]), float(here[2])), obj["box_min"], obj["box_max"])
    res["from_m"] = here.round(2).tolist()
    return res
