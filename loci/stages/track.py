"""Stage 4: SAM 3.1 finds every vocabulary phrase in every redacted keyframe.

Image mode per keyframe (one encode, then ~0.05 s per phrase). The video tracker drops objects on sparse
keyframes (see docs/benchmarks.md), so identity across frames is resolved later, in 3D (associate stage).
On the street, people and vehicles are also segmented, flagged `exclude`, and only used to keep them out
of the point cloud.
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from loci.masks import encode, suppress_across_labels
from loci.session import Session

log = logging.getLogger(__name__)


def run(
    session: Session,
    phrases: list[str],
    exclude_phrases: list[str],
    checkpoint: str,
    min_score: float = 0.5,
    min_area_frac: float = 0.0005,
) -> dict:
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=min_score)
    load_s = time.perf_counter() - t0

    instances = []
    kfs = session.keyframes()
    for kf in kfs:
        img = Image.open(session.keyframes_dir / kf["file"]).convert("RGB")
        h, w = img.height, img.width
        dets, excluded = [], []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(img)
            for phrase in [*phrases, *exclude_phrases]:
                out = proc.set_text_prompt(state=state, prompt=phrase)
                masks = out["masks"].squeeze(1).cpu().numpy().astype(bool)
                scores = out["scores"].float().cpu().numpy()
                for m, sc in zip(masks, scores, strict=True):
                    if m.sum() < min_area_frac * h * w:
                        continue
                    d = {"label": phrase, "score": float(sc), "mask": m}
                    (excluded if phrase in exclude_phrases else dets).append(d)
        for j, d in enumerate([*suppress_across_labels(dets), *({**e, "aliases": []} for e in excluded)]):
            ys, xs = np.nonzero(d["mask"])
            instances.append(
                {
                    "obs_id": f"obs_{kf['index']:04d}_{j:02d}",
                    "keyframe": kf["index"],
                    "t": kf["t"],
                    "label": d["label"],
                    "aliases": d["aliases"],
                    "score": round(d["score"], 3),
                    "exclude": d["label"] in exclude_phrases,
                    "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                    "area": int(d["mask"].sum()),
                    "image_size": [w, h],
                    "mask": encode(d["mask"]),
                }
            )

    seconds = time.perf_counter() - t0
    result = {
        "phrases": phrases,
        "exclude_phrases": exclude_phrases,
        "min_score": min_score,
        "seconds": round(seconds, 1),
        "load_s": round(load_s, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "instances": instances,
    }
    session.write(session.tracks_json, result)
    n_obj = sum(not i["exclude"] for i in instances)
    log.info(
        "%s: %d object instances (+%d excluded) over %d keyframes in %.0fs",
        session.id,
        n_obj,
        len(instances) - n_obj,
        len(kfs),
        seconds,
    )
    return result


def overlay_video(session: Session, out_path, fps: float = 2.0) -> None:
    """Debug overlay (M5 check): masks with labels on every keyframe."""
    from loci.masks import decode

    data = session.read(session.tracks_json)
    by_frame: dict[int, list] = {}
    for inst in data["instances"]:
        by_frame.setdefault(inst["keyframe"], []).append(inst)
    writer = None
    labels = sorted({i["label"] for i in data["instances"]})
    palette = {
        lab: np.array(
            cv2.applyColorMap(np.uint8([[int(255 * k / max(1, len(labels) - 1))]]), cv2.COLORMAP_TURBO)[0, 0],
            float,
        )
        for k, lab in enumerate(labels)
    }
    for kf in session.keyframes():
        img = cv2.imread(str(session.keyframes_dir / kf["file"])).astype(float)
        for inst in by_frame.get(kf["index"], []):
            m = decode(inst["mask"])
            img[m] = 0.5 * img[m] + 0.5 * palette[inst["label"]]
        img = img.astype(np.uint8)
        for inst in by_frame.get(kf["index"], []):
            x0, y0 = inst["bbox"][:2]
            txt = f"{inst['label']} {inst['score']:.2f}"
            cv2.putText(img, txt, (x0 + 2, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(img, txt, (x0 + 2, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        if writer is None:
            writer = cv2.VideoWriter(
                str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (img.shape[1], img.shape[0])
            )
        writer.write(img)
    if writer is not None:
        writer.release()
