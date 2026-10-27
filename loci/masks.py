"""Mask helpers: COCO RLE encode/decode, IoU, cross-label suppression."""

from __future__ import annotations

from typing import Any

import numpy as np
from pycocotools import mask as mask_util


def encode(mask: np.ndarray) -> dict[str, Any]:
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(x) for x in rle["size"]], "counts": rle["counts"].decode("ascii")}


def decode(rle: dict[str, Any]) -> np.ndarray:
    return mask_util.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(bool)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def suppress_across_labels(dets: list[dict[str, Any]], thresh: float = 0.85) -> list[dict[str, Any]]:
    """Greedy NMS over masks regardless of label; a suppressed detection's label becomes an alias.

    Each det needs "mask" (bool array), "score", "label". Returns kept dets with an "aliases" list.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i]["score"])
    kept: list[dict[str, Any]] = []
    for i in order:
        d = dets[i]
        for k in kept:
            if iou(d["mask"], k["mask"]) >= thresh:
                if d["label"] != k["label"] and d["label"] not in k["aliases"]:
                    k["aliases"].append(d["label"])
                break
        else:
            kept.append({**d, "aliases": []})
    return kept
