"""Evidence images: a redacted keyframe with the object's mask outlined, cropped around it (cached on disk)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from loci.masks import decode
from loci.session import Session

ACCENT = (80, 200, 255)  # BGR amber, matches the viewer highlight


@lru_cache(maxsize=16)
def _instances(tracks_path: str, mtime: float) -> dict[str, dict]:
    import json

    return {i["obs_id"]: i for i in json.loads(Path(tracks_path).read_text())["instances"]}


def render(session: Session, obs_id: str, full: bool = False, pad: float = 0.6) -> Path:
    out_dir = session.root / "evidence"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{obs_id}{'_full' if full else ''}.jpg"
    if out.exists():
        return out
    inst = _instances(str(session.tracks_json), session.tracks_json.stat().st_mtime).get(obs_id)
    if inst is None:
        raise KeyError(obs_id)
    img = cv2.imread(str(session.keyframes_dir / f"{inst['keyframe']:04d}.jpg"))  # redacted frame only
    mask = decode(inst["mask"]).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thick = max(2, round(max(img.shape[:2]) / 400))
    cv2.drawContours(img, contours, -1, (0, 0, 0), thick + 2)
    cv2.drawContours(img, contours, -1, ACCENT, thick)
    if not full:
        h, w = img.shape[:2]
        x0, y0, x1, y1 = inst["bbox"]
        bw, bh = x1 - x0, y1 - y0
        side = max(bw, bh) * (1 + 2 * pad)
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        side = max(side, 0.25 * min(w, h))
        xa, ya = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
        xb, yb = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
        img = img[ya:yb, xa:xb]
    cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return out
