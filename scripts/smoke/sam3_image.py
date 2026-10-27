"""Smoke test: SAM 3.1 image mode with text prompts. Reports load time, per-prompt latency and peak VRAM."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

log = logging.getLogger("smoke.sam3_image")

COLORS = np.array([[255, 80, 80], [80, 200, 255], [120, 255, 120], [255, 200, 60], [200, 120, 255]])


def overlay(img: np.ndarray, masks: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    for i, m in enumerate(masks):
        out[m] = 0.5 * out[m] + 0.5 * COLORS[i % len(COLORS)]
    return out.astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/sam3.1/sam3.1_multiplex.pt")
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--prompts", nargs="+", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="debug/smoke/sam3_image")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = build_sam3_image_model(checkpoint_path=args.checkpoint, load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=args.threshold)
    load_s = time.perf_counter() - t0
    log.info("loaded in %.1fs", load_s)

    results = []
    for path in args.images:
        image = Image.open(path).convert("RGB")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            t1 = time.perf_counter()
            state = proc.set_image(image)
            enc_s = time.perf_counter() - t1
            for prompt in args.prompts:
                t2 = time.perf_counter()
                out = proc.set_text_prompt(state=state, prompt=prompt)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t2
                masks = out["masks"].squeeze(1).cpu().numpy().astype(bool)
                scores = out["scores"].float().cpu().numpy().round(3).tolist()
                results.append(
                    {
                        "image": path,
                        "prompt": prompt,
                        "n": len(scores),
                        "scores": scores,
                        "encode_s": round(enc_s, 3),
                        "prompt_s": round(dt, 3),
                    }
                )
                log.info("%s | %-20s n=%d scores=%s %.3fs", Path(path).name, prompt, len(scores), scores, dt)
                if len(masks):
                    vis = overlay(np.array(image), masks)
                    name = f"{Path(path).stem}_{prompt.replace(' ', '_')}.jpg"
                    Image.fromarray(vis).save(out_dir / name)

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    summary = {"load_s": round(load_s, 1), "peak_vram_gb": round(peak_gb, 2), "results": results}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("peak VRAM %.2f GB", peak_gb)


if __name__ == "__main__":
    main()
