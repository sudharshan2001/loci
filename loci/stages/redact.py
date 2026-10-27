"""Stage 2: blur faces, plates, screens and documents with SAM 3.1 before anything is persisted.

Reads raw keyframes from raw_tmp/, writes redacted JPEGs to keyframes/, deletes raw_tmp/ (unless keep_raw),
and records per-prompt counts in privacy_ledger.json. Every later stage reads keyframes/ only.
"""

from __future__ import annotations

import logging
import shutil
import time

import cv2
import numpy as np

from loci.session import Session

log = logging.getLogger(__name__)


def blur_regions(img: np.ndarray, mask: np.ndarray, strength: float = 0.04, grow: float = 0.01) -> np.ndarray:
    """Return img with the masked area heavily blurred and a soft edge.

    strength: Gaussian sigma as a fraction of the longer image side.
    grow: mask dilation as a fraction of the longer image side, so edges of faces/plates are covered too.
    """
    if not mask.any():
        return img.copy()
    side = max(img.shape[:2])
    sigma = max(3.0, strength * side)
    # pixelate first so even a weak blur leaves no readable detail, then smooth
    small = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
    coarse = cv2.resize(small, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    blurred = cv2.GaussianBlur(coarse, (0, 0), sigma)
    r = max(1, round(grow * side))
    grown = cv2.dilate(
        mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    )
    alpha = cv2.GaussianBlur(grown.astype(np.float32), (0, 0), r / 2 + 1)
    alpha = np.maximum(alpha, grown)[..., None]  # fully blurred inside the grown mask, soft only outside it
    return (alpha * blurred + (1 - alpha) * img).astype(img.dtype)


def run(session: Session, prompts: dict[str, float], checkpoint: str, keep_raw: bool = False) -> dict:
    """prompts maps each text prompt to its confidence threshold (low = favour recall)."""
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    raw_files = sorted(session.raw_dir.glob("*.jpg"))
    if not raw_files:
        raise FileNotFoundError(f"no raw keyframes in {session.raw_dir}; run extract first")
    session.keyframes_dir.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = build_sam3_image_model(checkpoint_path=checkpoint, load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=min(prompts.values()))
    load_s = time.perf_counter() - t0

    ledger: dict[str, dict] = {p: {"instances": 0, "frames": 0} for p in prompts}
    per_frame = []
    for path in raw_files:
        bgr = cv2.imread(str(path))
        image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        union = np.zeros(bgr.shape[:2], bool)
        counts = {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(image)
            for prompt, thresh in prompts.items():
                out = proc.set_text_prompt(state=state, prompt=prompt)
                scores = out["scores"].float().cpu().numpy()
                keep = scores >= thresh
                masks = out["masks"].squeeze(1).cpu().numpy().astype(bool)[keep]
                if len(masks):
                    union |= masks.any(0)
                    ledger[prompt]["instances"] += len(masks)
                    ledger[prompt]["frames"] += 1
                counts[prompt] = int(len(masks))
        cv2.imwrite(
            str(session.keyframes_dir / path.name), blur_regions(bgr, union), [cv2.IMWRITE_JPEG_QUALITY, 92]
        )
        per_frame.append(
            {"file": path.name, "counts": counts, "blurred_fraction": round(float(union.mean()), 4)}
        )

    seconds = time.perf_counter() - t0
    result = {
        "prompts": prompts,
        "totals": ledger,
        "frames": per_frame,
        "n_keyframes": len(raw_files),
        "seconds": round(seconds, 1),
        "load_s": round(load_s, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
    }
    session.write(session.ledger, result)
    if not keep_raw:
        shutil.rmtree(session.raw_dir)
    log.info(
        "redacted %d keyframes in %.1fs: %s",
        len(raw_files),
        seconds,
        {p: v["instances"] for p, v in ledger.items()},
    )
    return result
