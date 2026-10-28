"""Compute and store capture quality for a session (used by `loci coach` and the API)."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from loci.quality import coach, reconstruction
from loci.session import Session


def run(session: Session) -> dict[str, Any]:
    meta = session.read(session.meta)
    kf = session.read(session.keyframes_json)
    recon = None
    if session.geometry.exists():
        g = np.load(session.geometry)
        times = np.array([k["t"] for k in kf["keyframes"]])[: len(g["poses"])]
        recon = reconstruction(times, g["poses"][:, :3, 3], meta.get("scene", "home"))
    attrs = session.read(session.attributes) if session.attributes.exists() else []
    utts = session.read(session.utterances) if session.utterances.exists() else None
    first = next(iter(sorted(session.keyframes_dir.glob("*.jpg"))), None)
    w, h = Image.open(first).size if first else (0, 0)
    q = {"reconstruction": recon, "tips": coach(meta, kf, recon, attrs, utts, w, h)}
    meta["quality"] = q
    session.write(session.meta, meta)
    return q
